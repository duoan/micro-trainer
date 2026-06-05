"""Module 3 - Evolution 4: Interleaved 1F1B (Megatron "virtual pipeline").

Principle in one line (Megatron-LM):
    Plain 1F1B still leaves a bubble of (p-1)/(m+p-1). The bubble shrinks if the pipeline
    is "deeper", but you only have p physical ranks. Interleaved 1F1B gives each rank V
    *virtual* chunks instead of one contiguous stage, so the pipeline behaves as if it had
    p*V stages. Layer-to-rank assignment is round-robin:

        global virtual stage s (0..p*V-1) lives on rank (s % p), as that rank's chunk (s // p).

    So rank r owns chunks {r, r+p, r+2p, ...}. A micro-batch flows through all p*V virtual
    stages, bouncing across the ranks V times. The schedule interleaves forwards of the
    different chunks during warmup so steady state is reached sooner -> the bubble shrinks
    by ~1/V (at the cost of V x more pipeline point-to-point messages).

Your battle zone:
    - `interleaved_schedule`: the warmup / steady / cooldown state machine that advances
      V virtual chunks per rank. Log [c1:F3] style tags and record Timeline spans so the
      Gantt shows a smaller bubble than one_forward_backward.py.

Run: python 3_pipeline_parallel/interleaved_1f1b.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: E402
import torch.nn as nn

from env_setup import (  # noqa: E402
    COMM_DEVICE,
    C,
    Timeline,
    launch_teaching_cluster,
    rank0_print,
    rank_print,
    render_gantt,
)

WORLD_SIZE = 4
NUM_CHUNKS = 2  # V: virtual chunks per rank (so p*V = 8 virtual stages)
NUM_MICRO = 8
MICRO_BS = 8
DIM = 32


class Chunk(nn.Module):
    """One virtual stage. Each rank owns NUM_CHUNKS of these."""

    def __init__(self, is_last_global: bool) -> None:
        super().__init__()
        self.fc = nn.Linear(DIM, DIM)
        self.is_last_global = is_last_global

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        return h if self.is_last_global else torch.relu(h)


def send_tensor(t: torch.Tensor, dst: int) -> None:
    dist.send(t.detach().to(COMM_DEVICE).contiguous(), dst=dst)


def recv_tensor(shape: tuple[int, ...], src: int, device: torch.device) -> torch.Tensor:
    buf = torch.empty(shape, device=COMM_DEVICE)
    dist.recv(buf, src=src)
    return buf.to(device)


def log_state(rank: int, chunk: int, tag: str) -> None:
    """Live conveyor log. tag like 'F3'/'B1'; chunk is the virtual-chunk index on this rank."""
    color = C.CYAN if tag.startswith("F") else C.MAGENTA
    print(f"{C.rank(rank)}rank{rank}{C.RESET} {color}[c{chunk}:{tag}]{C.RESET}", flush=True)


def global_stage(rank: int, chunk: int, world_size: int) -> int:
    """Map (rank, virtual chunk) -> global virtual-stage index in 0..world_size*NUM_CHUNKS-1."""
    return chunk * world_size + rank


def interleaved_schedule(
    rank: int,
    world_size: int,
    chunks: list[Chunk],
    micro_inputs: list[torch.Tensor] | None,
    device: torch.device,
    timeline: Timeline,
) -> None:
    """Interleaved 1F1B over NUM_CHUNKS virtual stages per rank.

    Pipeline depth is now p*V. A micro-batch visits, in order, global stages
    0,1,...,p*V-1, i.e. it alternates ranks and climbs through the chunks.

    ============================ YOUR BATTLE ZONE ============================
    Treat the pipeline as having `p*V` stages. Reuse the 1F1B idea but pick WHICH
    virtual chunk to run on each step (Megatron uses a deterministic interleave order).

    A workable plan:
      - num_warmup = min((world_size - 1 - rank) + (NUM_CHUNKS - 1) * world_size, total_fwds)
        (deeper virtual pipeline => longer warmup).
      - Maintain, per virtual chunk, a queue of forwarded-but-not-yet-backward activations.
      - Warmup: forward-only, cycling chunk = (step // 1) pattern Megatron uses (advance the
        lowest chunk first, then the next), sending each chunk's output to the rank that owns
        the NEXT global stage = next_rank(global_stage+1).
      - Steady: one forward (next chunk in order) immediately followed by one backward
        (the oldest pending activation, in reverse chunk order).
      - Cooldown: drain remaining backwards.
      Use global_stage(rank, chunk, world_size) to know if a chunk is the first/last global
      stage (first => read micro_inputs / no recv; last => start backward from a local loss).
      Call log_state(rank, chunk, f"F{m}") / timeline.span(rank, f"c{chunk}F{m}") for each chunk.
    ==========================================================================
    """
    # TODO(you): interleaved warmup / steady / cooldown over NUM_CHUNKS virtual chunks
    raise NotImplementedError("TODO: interleaved_schedule -- hand-write the virtual-pipeline 1F1B state machine")


def run(rank: int, world_size: int, device: torch.device) -> None:
    torch.manual_seed(100 + rank)
    total_stages = world_size * NUM_CHUNKS
    chunks = [
        Chunk(is_last_global=(global_stage(rank, c, world_size) == total_stages - 1)).to(device)
        for c in range(NUM_CHUNKS)
    ]

    micro_inputs = None
    if rank == 0:  # rank 0 owns global stage 0 (chunk 0), the pipeline entry
        g = torch.Generator().manual_seed(2024)
        micro_inputs = [torch.randn(MICRO_BS, DIM, generator=g).to(device) for _ in range(NUM_MICRO)]

    rank_print(rank, f"owns virtual chunks {[global_stage(rank, c, world_size) for c in range(NUM_CHUNKS)]} of {total_stages}")

    timeline = Timeline()
    interleaved_schedule(rank, world_size, chunks, micro_inputs, device, timeline)

    rank0_print(rank, render_gantt(timeline, world_size))
    rank0_print(rank, "Interleaved 1F1B: V virtual chunks per rank shrink the bubble ~1/V. Compare with one_forward_backward.py!")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
