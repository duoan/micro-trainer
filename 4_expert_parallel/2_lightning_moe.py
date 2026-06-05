"""Module 4 (Expert Parallelism) - Evolution: Lightning MoE (Tile-level micro-pipeline, full comm-compute overlap).

Principle in one line (the soul of MegaScale-MoE):
    naive_moe's problem is "send ALL tokens first, then compute everything" -- comm and
    compute are cleanly separated. Lightning's idea: split the whole token batch into
    several TILES and let them pipeline -- while tile t's expert is COMPUTING, tile t+1's
    All-to-All is already FLYING asynchronously over the NIC. By the time the compute
    finishes, the next tile's data has roughly arrived. The communication latency gets
    HIDDEN behind compute, so total time ~= max(total comm, total compute), not their sum.

    This is Tile Overlap: async All-to-All + a software pipeline fill the bubble.

    Compare against naive's makespan: Lightning should be markedly shorter (the larger
    the latency and the more tiles, the more dramatic the gap).

Your battle zone:
    - `lightning_moe_forward`: use slow_all_to_all_single_async for dispatch, building a
      "fire next tile's comm / compute current tile" micro-pipeline.
    - Finally use env_setup.bar to draw an ASCII performance comparison bar (naive vs
      lightning), wrapping up selling-point #4.

Run: python 4_expert_parallel/lightning_moe.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from topology_sim import (  # noqa: F401  (async All-to-All, for overlap)
    DEFAULT_LINK,
    slow_all_to_all_single_async,
)

from env_setup import (
    COMM_DEVICE,  # noqa: F401  (remember to move comm tensors onto it)
    Timeline,
    bar,
    launch_teaching_cluster,
    rank0_print,
    rank_print,
)

WORLD_SIZE = 4
TOKENS_PER_EXPERT = 32
DIM = 64
NUM_TILES = 4  # how many tiles to split tokens along the batch dim (more = smoother pipeline, but more per-tile overhead)


class Expert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def make_local_tokens(rank: int, world_size: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator().manual_seed(7000 + rank)
    return torch.randn(world_size * TOKENS_PER_EXPERT, DIM, generator=g).to(device)


def split_tiles(tokens: torch.Tensor, num_tiles: int) -> list[torch.Tensor]:
    """Split the whole batch into num_tiles tiles along the token dim. Each tile keeps
    the "segmented by expert" layout.

    To keep All-to-All's equal-split valid on every tile, we split by a divisor of
    TOKENS_PER_EXPERT (guaranteeing each tile has equal rows destined for each expert).
    """
    assert tokens.shape[0] % num_tiles == 0, "token count must be divisible by NUM_TILES"
    return list(tokens.chunk(num_tiles, dim=0))


def lightning_moe_forward(
    rank: int,
    world_size: int,
    expert: Expert,
    tokens: torch.Tensor,
    device: torch.device,
    timeline: Timeline,
) -> torch.Tensor:
    """Tile-level micro-pipeline MoE: async dispatch overlapped with expert compute.

    ============================ YOUR BATTLE ZONE ============================
    Idea (software pipeline, prologue / steady / epilogue):

      tiles = split_tiles(tokens, NUM_TILES)

      1) Prologue: fire an async dispatch for tile 0:
             buf0 = empty_like(tile0) on COMM_DEVICE
             work0 = slow_all_to_all_single_async(buf0, tile0)

      2) Steady: for t in range(NUM_TILES):
             - if there is a next tile (t+1), immediately fire its async dispatch (get comm flying)
             - work_t.wait()            # wait for the current tile's data (time as kind="comm")
             - out_t = expert(buf_t)    # compute the current tile (time as kind="compute")
                                        # KEY: this compute exactly "covers" tile t+1's comm latency
             - stash out_t (combine can be pipelined the same way; get dispatch overlap working first)

      3) Epilogue: torch.cat all out_t, optionally do one more combine back, return the result.

    Contrast: in naive, wait() immediately follows send (nothing to do); here, before
    wait() we have already fired the next tile's comm and are about to do a big chunk of
    compute, so the latency gets eaten by compute = overlap.
    ==========================================================================
    """
    # TODO(you): use async all_to_all to build a tile micro-pipeline so comm overlaps compute
    raise NotImplementedError("TODO: lightning_moe_forward -- hand-write the Tile-level async comm-compute overlap pipeline")


def print_compare_bar(rank: int, naive_ms: float, lightning_ms: float) -> None:
    """Selling-point #4 wrap-up: draw an ASCII comparison bar of naive vs lightning total time.

    naive_ms here is a placeholder; the real usage is to plug in the makespan that
    naive_moe.py printed and compare.
    """
    if rank != 0:
        return
    worst = max(naive_ms, lightning_ms, 1.0)
    print()
    print("  -- Comm-bubble fill effect (shorter is better) --")
    print(f"  naive     |{bar(naive_ms, worst)}| {naive_ms:6.1f} ms")
    print(f"  lightning |{bar(lightning_ms, worst)}| {lightning_ms:6.1f} ms")
    if lightning_ms < naive_ms:
        saved = (1 - lightning_ms / naive_ms) * 100
        print(f"  Tile Overlap saved ~{saved:.0f}% of the time -- comm latency hidden behind compute!")


def run(rank: int, world_size: int, device: torch.device) -> None:
    torch.manual_seed(300 + rank)
    expert = Expert().to(device)
    tokens = make_local_tokens(rank, world_size, device)

    rank_print(rank, f"local tokens = {tuple(tokens.shape)} | NUM_TILES = {NUM_TILES}")

    timeline = Timeline()
    _ = lightning_moe_forward(rank, world_size, expert, tokens, device, timeline)

    lightning_ms = timeline.makespan() * 1000
    rank0_print(rank, f"lightning MoE makespan = {lightning_ms:.1f} ms (comm overlapped with compute)")

    # Plug the makespan that naive_moe.py printed below to compare (placeholder shown here).
    naive_ms_placeholder = lightning_ms * 1.8
    print_compare_bar(rank, naive_ms_placeholder, lightning_ms)


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
