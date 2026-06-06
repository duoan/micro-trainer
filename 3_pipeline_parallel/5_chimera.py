"""Module 3 - Evolution 6: Chimera (bidirectional pipeline, two combined directions).

Principle in one line (Chimera, SC'21 -- the academic predecessor of DeepSeek DualPipe):
    A single 1F1B pipeline only flows one way, so its bubble cannot drop below (p-1)/(m+p-1).
    Chimera runs TWO pipelines at once in OPPOSITE directions, and makes each rank hold BOTH
    a "down" stage and an "up" stage:

        down pipeline: stage s on rank s          (flows rank 0 -> p-1)
        up   pipeline: stage s on rank (p-1 - s)  (flows rank p-1 -> 0)

    So rank r owns down-stage r and up-stage (p-1-r). The micro-batches are split into two
    groups, one fed into each pipeline. While the down pipeline's bubble would idle a rank,
    the up pipeline gives that same rank work (and vice versa), so the combined bubble is
    roughly halved. The two stage copies on each rank are kept in sync by All-Reducing their
    gradients before the optimizer step (they are replicas of the same logical layer).

Your battle zone:
    - `chimera_schedule`: drive the two opposing 1F1B pipelines so each rank interleaves its
      down-stage and up-stage work; record Timeline spans showing the halved bubble, then
      All-Reduce the two replicas' grads. Compare with 6_deepseek_dualpipe.py (its successor).

Run: python 3_pipeline_parallel/5_chimera.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: E402
import torch.nn as nn

from env_setup import (  # noqa: E402
    C,
    Timeline,
    launch_teaching_cluster,
    rank0_print,
    rank_print,
    render_gantt,
)

WORLD_SIZE = 4  # = pipeline depth p
NUM_MICRO = 4  # micro-batches PER direction
MICRO_BS = 8
DIM = 32


class Stage(nn.Module):
    def __init__(self, is_tail: bool) -> None:
        super().__init__()
        self.fc = nn.Linear(DIM, DIM)
        self.is_tail = is_tail

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        return h if self.is_tail else torch.relu(h)


def send_tensor(t: torch.Tensor, dst: int) -> None:
    dist.send(t.detach().contiguous(), dst=dst)


def recv_tensor(shape: tuple[int, ...], src: int, device: torch.device) -> torch.Tensor:
    buf = torch.empty(shape, device=device)
    dist.recv(buf, src=src)
    return buf.to(device)


def log_state(rank: int, tag: str) -> None:
    """Live conveyor log. tag like 'D:F2' (down pipeline) / 'U:B1' (up pipeline)."""
    color = C.CYAN if ":F" in tag else C.MAGENTA
    print(f"{C.rank(rank)}rank{rank}{C.RESET} {color}[{tag}]{C.RESET}", flush=True)


def sync_replica_grads(down_stage: Stage, up_stage: Stage, world_size: int) -> None:
    """Provided helper: the two stage copies on this rank are replicas; average their grads.

    (In real Chimera the matching down/up stages live on different ranks and are synced with
    an All-Reduce across those ranks. Here both copies are local, so we just average them and
    All-Reduce across the world to mimic the replica sync -- safe to call after backward.)
    """
    for p_down, p_up in zip(down_stage.parameters(), up_stage.parameters(), strict=True):
        if p_down.grad is None or p_up.grad is None:
            continue
        g = p_down.grad + p_up.grad
        dist.all_reduce(g, op=dist.ReduceOp.SUM)
        g = g / (2 * world_size)
        p_down.grad, p_up.grad = g.clone(), g.clone()


def chimera_schedule(
    rank: int,
    world_size: int,
    down_stage: Stage,
    up_stage: Stage,
    micro_down: list[torch.Tensor] | None,
    micro_up: list[torch.Tensor] | None,
    device: torch.device,
    timeline: Timeline,
) -> None:
    """Two opposing 1F1B pipelines combined on the same ranks.

    Down pipeline: forward rank r -> r+1, backward r <- r+1   (uses down_stage).
    Up   pipeline: forward rank r -> r-1, backward r <- r+1   (uses up_stage).
    rank 0 feeds micro_down; rank world_size-1 feeds micro_up.

    ============================ YOUR BATTLE ZONE ============================
    Run BOTH pipelines as interleaved 1F1B state machines on this rank:
      - down warmup = world_size - 1 - rank ; up warmup = rank.
      - Maintain a pending queue per direction (activations awaiting backward).
      - Steady: alternate a down step and an up step so the rank is busy whenever EITHER
        pipeline has work -- that overlap is what halves the bubble.
      - Tail stages: down pipeline's tail is rank world_size-1; up pipeline's tail is rank 0.
        The tail builds a local loss to start its backward.
      Use log_state(rank, 'D:F{m}') / 'U:B{m}' and timeline.span(rank, tag) for each chunk.
    After both pipelines drain, call sync_replica_grads(down_stage, up_stage, world_size).
    ==========================================================================
    """
    # TODO(you): interleave the down and up 1F1B pipelines, then sync the two replicas' grads
    raise NotImplementedError("TODO: chimera_schedule -- hand-write the two combined opposing pipelines")


def run(rank: int, world_size: int, device: torch.device) -> None:
    torch.manual_seed(100 + rank)
    down_stage = Stage(is_tail=(rank == world_size - 1)).to(device)
    up_stage = Stage(is_tail=(rank == 0)).to(device)

    micro_down = micro_up = None
    if rank == 0:
        gd = torch.Generator().manual_seed(2024)
        micro_down = [torch.randn(MICRO_BS, DIM, generator=gd).to(device) for _ in range(NUM_MICRO)]
    if rank == world_size - 1:
        gu = torch.Generator().manual_seed(4048)
        micro_up = [torch.randn(MICRO_BS, DIM, generator=gu).to(device) for _ in range(NUM_MICRO)]

    rank_print(rank, f"holds down-stage {rank} and up-stage {world_size - 1 - rank}")

    timeline = Timeline()
    chimera_schedule(rank, world_size, down_stage, up_stage, micro_down, micro_up, device, timeline)

    rank0_print(rank, render_gantt(timeline, world_size))
    rank0_print(rank, "Chimera: two opposing pipelines halve the bubble; DeepSeek DualPipe is its descendant.")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
