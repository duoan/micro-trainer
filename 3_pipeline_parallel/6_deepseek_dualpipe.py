"""Module 3 (DeepSeek) - DualPipe: bidirectional pipeline with compute/comm overlap.

Principle in one line (DeepSeek-V3):
    1F1B still leaves a (p-1)/(m+p-1) bubble and runs one direction. DualPipe feeds TWO
    micro-batch streams from OPPOSITE ENDS of the pipeline at the same time: stream A flows
    stage 0 -> p-1, stream B flows stage p-1 -> 0. Each stage thus always has work from one
    direction to fill what would be the other direction's bubble. Combined with overlapping
    the pipeline point-to-point comm with compute, the bubble nearly vanishes -- at the cost
    of keeping two copies of the parameters/activations in flight.

    DeepSeek-V3 pairs DualPipe with MoE: the all-to-all expert comm of one chunk is hidden
    behind the attention/MLP compute of another chunk. This demo focuses on the SCHEDULING
    skeleton (the bidirectional 1F1B-style state machine) and visualizes it on the Timeline.

Your battle zone:
    - `dualpipe_schedule`: run the two opposing streams, interleaving forward/backward so each
      stage stays busy; record spans so render_gantt shows the bubble shrink vs 1F1B/GPipe.

Run: python 3_pipeline_parallel/6_deepseek_dualpipe.py
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

WORLD_SIZE = 4  # = number of pipeline stages
NUM_MICRO = 6   # micro-batches PER direction
MICRO_BS = 8
DIM = 32


class Stage(nn.Module):
    """One pipeline stage. In DualPipe each rank holds the stage for BOTH directions
    (here we share one Stage module for simplicity; real DualPipe keeps separate params)."""

    def __init__(self, is_edge: bool) -> None:
        super().__init__()
        self.fc = nn.Linear(DIM, DIM)
        self.is_edge = is_edge

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        return h if self.is_edge else torch.relu(h)


def send_tensor(t: torch.Tensor, dst: int) -> None:
    dist.send(t.detach().contiguous(), dst=dst)


def recv_tensor(shape: tuple[int, ...], src: int, device: torch.device) -> torch.Tensor:
    buf = torch.empty(shape, device=device)
    dist.recv(buf, src=src)
    return buf.to(device)


def log_state(rank: int, tag: str) -> None:
    """Live conveyor log. tag like 'A:F2' (stream A forward mb2) or 'B:B1' (stream B backward mb1)."""
    color = C.CYAN if ":F" in tag else C.MAGENTA
    print(f"{C.rank(rank)}rank{rank}{C.RESET} {color}[{tag}]{C.RESET}", flush=True)


def dualpipe_schedule(
    rank: int,
    world_size: int,
    stage: Stage,
    micro_inputs_a: list[torch.Tensor] | None,
    micro_inputs_b: list[torch.Tensor] | None,
    device: torch.device,
    timeline: Timeline,
) -> None:
    """Bidirectional pipeline schedule.

    Two streams:
        Stream A: forward direction rank r -> r+1 (rank 0 holds its inputs).
        Stream B: forward direction rank r -> r-1 (rank world_size-1 holds its inputs).
    So each rank receives stream-A activations from rank-1 and stream-B activations from rank+1,
    and can almost always overlap one stream's compute with the other stream's comm.

    ============================ YOUR BATTLE ZONE ============================
    Mirror the 1F1B state machine, but for TWO directions at once. A workable plan:

    1) Warmup: for each direction, do forward-only for its warmup count to fill the pipe
       (stream A warmup = world_size-1-rank, stream B warmup = rank). Use log_state with the
       'A:F{m}' / 'B:F{m}' tags and timeline.span(rank, tag) for each compute chunk.

    2) Steady: each iteration, advance BOTH streams by one (a forward and a backward per
       direction), interleaving so that while stream A's activation is being sent to rank+1,
       stream B's chunk is computing (and vice versa). This interleave is what fills the bubble.

    3) Cooldown: drain the remaining backwards of both streams.

    Helpers to write inline: fwd_a / bwd_a (use send_tensor(.., rank+1) / recv from rank-1) and
    fwd_b / bwd_b (use send_tensor(.., rank-1) / recv from rank+1). Edge stages (rank 0 for B,
    rank world_size-1 for A) start from their own inputs and a local loss for backward.

    Tip: keep this teaching version readable -- correctness of the *schedule shape* (both
    streams complete, timeline bubble smaller than 1F1B) matters more than perfect numerics.
    ==========================================================================
    """
    # TODO(you): bidirectional warmup / steady / cooldown, interleaving stream A and stream B
    raise NotImplementedError("TODO: dualpipe_schedule -- hand-write the bidirectional pipeline state machine")


def run(rank: int, world_size: int, device: torch.device) -> None:
    torch.manual_seed(100 + rank)
    is_edge = rank in (0, world_size - 1)
    stage = Stage(is_edge=is_edge).to(device)

    # Stream A originates at rank 0; stream B originates at the last rank.
    micro_inputs_a = micro_inputs_b = None
    if rank == 0:
        ga = torch.Generator().manual_seed(2024)
        micro_inputs_a = [torch.randn(MICRO_BS, DIM, generator=ga).to(device) for _ in range(NUM_MICRO)]
    if rank == world_size - 1:
        gb = torch.Generator().manual_seed(4048)
        micro_inputs_b = [torch.randn(MICRO_BS, DIM, generator=gb).to(device) for _ in range(NUM_MICRO)]

    warmup_a = min(world_size - 1 - rank, NUM_MICRO)
    warmup_b = min(rank, NUM_MICRO)
    rank_print(rank, f"stage {rank} | streamA warmup = {warmup_a}, streamB warmup = {warmup_b}")

    timeline = Timeline()
    dualpipe_schedule(rank, world_size, stage, micro_inputs_a, micro_inputs_b, device, timeline)

    rank0_print(rank, render_gantt(timeline, world_size))
    rank0_print(rank, "DualPipe: two opposing streams fill each other's bubbles -- compare with 1_gpipe.py / 2_one_forward_backward.py.")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
