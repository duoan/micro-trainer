"""Module 3 - Baseline 3: GPipe scheduling (continuous micro-batch injection + big Bubble pain).

Principle in one line:
    Pipeline parallelism = split the model BY LAYERS into several stages, and each
    rank holds one stage (rank0 is the first layers...). A batch is split into M
    micro-batches that flow through the stages like an assembly line:
        rank0 finishes micro-batch 0's forward, sends the activation to rank1, then
        immediately starts micro-batch 1's forward...

    GPipe's schedule is the simplest: run ALL micro-batches' forwards first, then do
    all backwards (F-then-B). The cost is lots of idle time during the fill/drain
    phases -- the infamous "pipeline bubble". More ranks or fewer micro-batches make
    the bubble a larger fraction of the time.

    Look at the Timeline dashboard after running: you'll see clear staircase-shaped
    gaps (bubbles). The next lesson, 1F1B, fills them in.

Your battle zone:
    - `gpipe_schedule`: implement the "all forwards -> all backwards" schedule,
      hand-writing dist.send / dist.recv.

Run: python 3_pipeline_parallel/1_gpipe.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: E402
import torch.nn as nn

from env_setup import (  # noqa: E402
    Timeline,
    launch_teaching_cluster,
    rank0_print,
    rank_print,
    render_gantt,
)

WORLD_SIZE = 4  # = number of pipeline stages
NUM_MICRO = 4  # number of micro-batches
MICRO_BS = 8
DIM = 32


class Stage(nn.Module):
    """A single pipeline stage: one linear layer + activation. Each rank holds one."""

    def __init__(self, is_last: bool) -> None:
        super().__init__()
        self.fc = nn.Linear(DIM, DIM)
        self.is_last = is_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        return h if self.is_last else torch.relu(h)


# --------------------------------------------------------------------------- #
# Point-to-point comm helpers (send/recv run directly on the compute device)
# --------------------------------------------------------------------------- #
def send_tensor(t: torch.Tensor, dst: int) -> None:
    dist.send(t.detach().contiguous(), dst=dst)


def recv_tensor(shape: tuple[int, ...], src: int, device: torch.device) -> torch.Tensor:
    buf = torch.empty(shape, device=device)
    dist.recv(buf, src=src)
    return buf


def gpipe_schedule(
    rank: int,
    world_size: int,
    stage: Stage,
    micro_inputs: list[torch.Tensor] | None,
    device: torch.device,
    timeline: Timeline,
) -> torch.Tensor:
    """GPipe (F-then-B) schedule.

    Conventions:
        - rank 0 is the first stage and holds the real input micro_inputs (None elsewhere).
        - Forward direction rank r -> r+1; backward direction rank r <- r+1.
        - Wrap each forward/backward/comm segment with timeline.span(rank, label, kind=...)
          so render_gantt can draw the bubbles.

    ============================ YOUR BATTLE ZONE ============================
    Phase A -- all forwards (for m in range(NUM_MICRO)):
        * rank 0: x = micro_inputs[m]; otherwise x = recv_tensor(..., src=rank-1)
        * wrap out = stage(x) with timeline.span(rank, f"F{m}")
        * if not the last stage: send_tensor(out, dst=rank+1); else keep out for backward
        * remember to save each micro-batch's input/output (a list) -- backward needs them

    Phase B -- all backwards (iterate m, flowing in the opposite direction of forward):
        * last stage: start from a simple loss (e.g. out.pow(2).mean()), backprop to a grad
        * other stages: recv the gradient from rank+1, feed it via out.backward(grad)
        * if not the first stage: send the input's gradient to rank-1
        * likewise wrap with timeline.span(rank, f"B{m}")

    Hint: during forward, set requires_grad_(True) on tensors that need a returned
    gradient, or keep the autograd graph alive.
    ==========================================================================
    """
    # TODO(you): first implement "all forwards", then "all backwards", chaining adjacent stages with send/recv
    raise NotImplementedError("TODO: gpipe_schedule -- hand-write the F-then-B pipeline schedule (raw send/recv)")


def run(rank: int, world_size: int, device: torch.device) -> None:
    torch.manual_seed(100 + rank)  # each stage gets different parameters
    stage = Stage(is_last=(rank == world_size - 1)).to(device)

    micro_inputs = None
    if rank == 0:
        g = torch.Generator().manual_seed(2024)
        micro_inputs = [torch.randn(MICRO_BS, DIM, generator=g).to(device) for _ in range(NUM_MICRO)]

    rank_print(rank, f"I am stage {rank}/{world_size - 1} (is_last={rank == world_size - 1})")

    timeline = Timeline()
    gpipe_schedule(rank, world_size, stage, micro_inputs, device, timeline)

    # rank 0 draws the dashboard (simplified: drawing only its own row is fine;
    # advanced: all_gather each rank's events).
    rank0_print(rank, render_gantt(timeline, world_size))
    rank0_print(rank, "Note the staircase gaps: that's the GPipe Bubble. Next lesson 1F1B fills it in.")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
