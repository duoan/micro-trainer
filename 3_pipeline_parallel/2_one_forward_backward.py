"""Module 3 - Evolution 3: 1F1B scheduling (one-forward-one-backward, live state machine).

Principle in one line:
    GPipe piles up all forwards before any backward, which forces all activations to
    stay alive (memory blows up) plus a big bubble. The essence of 1F1B
    (One-Forward-One-Backward) is: once the pipeline reaches STEADY STATE, every
    stage, right after finishing "one micro-batch's forward", immediately does "one
    micro-batch's backward". Alternating like this -> activations freed promptly
    (stable memory) -> the bubble is largely filled by overlapping fill and drain.

    The schedule has three phases:
        1. Warmup: the first few micro-batches do forward only, to FILL the pipeline.
           warmup count = (world_size - 1 - rank). Earlier stages fill more.
        2. Steady: alternate 1F1B -- do one forward, then one backward.
        3. Cooldown: finish the backwards for the remaining in-flight micro-batches.

    Compare against the gpipe dashboard: 1F1B's compute blocks should pack tighter
    with fewer bubbles.

Your battle zone:
    - `one_f_one_b_schedule`: implement the warmup / steady / cooldown state machine,
      and print a live [F0][F1][B0]... state-machine log (the conveyor matrix).

Run: python 3_pipeline_parallel/2_one_forward_backward.py
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

WORLD_SIZE = 4
NUM_MICRO = 8  # more micro-batches so the steady-state advantage of 1F1B is visible
MICRO_BS = 8
DIM = 32


class Stage(nn.Module):
    def __init__(self, is_last: bool) -> None:
        super().__init__()
        self.fc = nn.Linear(DIM, DIM)
        self.is_last = is_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        return h if self.is_last else torch.relu(h)


def send_tensor(t: torch.Tensor, dst: int) -> None:
    dist.send(t.detach().contiguous(), dst=dst)


def recv_tensor(shape: tuple[int, ...], src: int, device: torch.device) -> torch.Tensor:
    buf = torch.empty(shape, device=device)
    dist.recv(buf, src=src)
    return buf.to(device)


def log_state(rank: int, tag: str) -> None:
    """Live state-machine log: print which micro-batch's F/B this stage is doing now.

    tag looks like "F3" / "B1". F uses one color, B another, so the conveyor rhythm
    is visible to the naked eye.
    """
    color = C.CYAN if tag.startswith("F") else C.MAGENTA
    print(f"{C.rank(rank)}rank{rank}{C.RESET} {color}[{tag}]{C.RESET}", flush=True)


def one_f_one_b_schedule(
    rank: int,
    world_size: int,
    stage: Stage,
    micro_inputs: list[torch.Tensor] | None,
    device: torch.device,
    timeline: Timeline,
) -> None:
    """The 1F1B state-machine schedule.

    Comm direction is the same as GPipe: forward r -> r+1, backward r <- r+1.
    warmup count: num_warmup = min(world_size - 1 - rank, NUM_MICRO)

    ============================ YOUR BATTLE ZONE ============================
    Maintain a stack/queue of "forwarded, awaiting-backward" micro-batch contexts
    (input tensor, output tensor).

    1) Warmup: loop num_warmup times doing forward only (forward_one), call
       log_state(rank, f"F{m}"), time with timeline.span(rank, f"F{m}"), and send the
       forward result to rank+1 as needed.

    2) Steady: loop (NUM_MICRO - num_warmup) times:
         - first forward_one (inject one more new micro-batch's forward)
         - then immediately backward_one (backward the oldest awaiting micro-batch)
       This forward-then-backward IS "1F1B".

    3) Cooldown: finish backward for all remaining micro-batches in the queue.

    Suggestion: write forward_one / backward_one as inner helpers:
       forward_one(m): get/recv input -> stage(x) -> time+log -> send or stash
       backward_one(m): get/recv grad -> out.backward(grad) -> time+log -> send input grad to rank-1
    ==========================================================================
    """
    # TODO(you): the three-phase warmup / steady / cooldown 1F1B state machine
    raise NotImplementedError("TODO: one_f_one_b_schedule -- hand-write the 3-phase 1F1B state machine (raw send/recv)")


def run(rank: int, world_size: int, device: torch.device) -> None:
    torch.manual_seed(100 + rank)
    stage = Stage(is_last=(rank == world_size - 1)).to(device)

    micro_inputs = None
    if rank == 0:
        g = torch.Generator().manual_seed(2024)
        micro_inputs = [torch.randn(MICRO_BS, DIM, generator=g).to(device) for _ in range(NUM_MICRO)]

    num_warmup = min(world_size - 1 - rank, NUM_MICRO)
    rank_print(rank, f"stage {rank} | warmup micro-batches = {num_warmup}")

    timeline = Timeline()
    one_f_one_b_schedule(rank, world_size, stage, micro_inputs, device, timeline)

    rank0_print(rank, render_gantt(timeline, world_size))
    rank0_print(rank, "1F1B: tighter compute blocks, fewer bubbles, stabler memory. Compare with 1_gpipe.py!")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
