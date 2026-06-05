"""Module 3 - Evolution 5: PipeDream (asynchronous pipeline + weight stashing).

Principle in one line (PipeDream, MSR):
    GPipe/1F1B are SYNCHRONOUS: every stage flushes and applies one global gradient before
    the next batch. PipeDream keeps the pipe ALWAYS full by never flushing -- each stage
    updates its weights as soon as a micro-batch's backward arrives. But that creates a
    consistency hazard: micro-batch m's forward used weight version v, yet by the time its
    backward runs the stage may already be on version v+2. Computing the gradient against
    the wrong weights is incorrect.

    PipeDream's fix is WEIGHT STASHING: when a stage does the forward for micro-batch m, it
    stashes the exact weight version it used. When m's backward comes back, it temporarily
    restores that stashed version, computes the gradient against it, then returns to the
    latest weights to keep stepping. This makes the async pipeline numerically equivalent to
    a consistent per-micro-batch update while keeping the bubble near zero.

Your battle zone:
    - `pipedream_schedule`: run steady-state 1F1B WITHOUT a global flush, and manage the
      per-micro-batch weight stash (stash on forward, restore on backward, then step).

Run: python 3_pipeline_parallel/pipedream.py
"""

from __future__ import annotations

import copy
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
NUM_MICRO = 8
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
    dist.send(t.detach().to(COMM_DEVICE).contiguous(), dst=dst)


def recv_tensor(shape: tuple[int, ...], src: int, device: torch.device) -> torch.Tensor:
    buf = torch.empty(shape, device=COMM_DEVICE)
    dist.recv(buf, src=src)
    return buf.to(device)


def log_state(rank: int, tag: str, version: int) -> None:
    """Live log. tag 'F3'/'B1' plus the weight VERSION used (the whole point of PipeDream)."""
    color = C.CYAN if tag.startswith("F") else C.MAGENTA
    print(f"{C.rank(rank)}rank{rank}{C.RESET} {color}[{tag}]{C.RESET}{C.GREY}(w_v{version}){C.RESET}", flush=True)


def stash_weights(stage: Stage) -> dict:
    """Provided helper: snapshot a stage's current weights (a weight 'version')."""
    return copy.deepcopy(stage.state_dict())


def restore_weights(stage: Stage, snapshot: dict) -> None:
    """Provided helper: temporarily load a stashed weight version back into the stage."""
    stage.load_state_dict(snapshot)


def pipedream_schedule(
    rank: int,
    world_size: int,
    stage: Stage,
    micro_inputs: list[torch.Tensor] | None,
    device: torch.device,
    timeline: Timeline,
) -> None:
    """Asynchronous 1F1B with weight stashing (no global flush).

    Same warmup count as 1F1B: num_warmup = min(world_size - 1 - rank, NUM_MICRO).
    Comm direction: forward r -> r+1, backward r <- r+1.

    ============================ YOUR BATTLE ZONE ============================
    Maintain:
      - `version`: an int counter of how many times THIS stage has stepped its optimizer.
      - `pending`: a queue of dicts per in-flight micro-batch, e.g.
            {"in": x, "out": y, "stash": stash_weights(stage), "ver": version}
        The stash is the weights that produced `out` -- needed for a correct backward.

    forward_one(m):
        x = micro_inputs[m] (rank 0) or recv from rank-1
        snapshot = stash_weights(stage)              # <-- stash BEFORE compute
        out = stage(x); log_state(rank, f"F{m}", version); timeline.span(...)
        push {in, out, stash: snapshot, ver: version} to `pending`; send out to rank+1 (if any)

    backward_one():
        item = pending.pop(0) (oldest)
        restore_weights(stage, item["stash"])        # <-- compute grad vs the SAME weights
        recv grad (or build local loss on last stage); item["out"].backward(grad)
        send input-grad to rank-1 (if any)
        restore the LATEST weights, take an SGD step on stage params, version += 1
        log_state(rank, f"B{m}", item["ver"])

    Phases: warmup (forward_one only), steady (forward_one then backward_one), cooldown
    (drain backward_one). The key teaching point is that backward uses item["stash"], NOT
    the current weights -- print the version to make the async-but-consistent behavior visible.
    ==========================================================================
    """
    # TODO(you): async 1F1B with per-micro-batch weight stashing (stash on F, restore on B)
    raise NotImplementedError("TODO: pipedream_schedule -- hand-write async 1F1B + weight stashing")


def run(rank: int, world_size: int, device: torch.device) -> None:
    torch.manual_seed(100 + rank)
    stage = Stage(is_last=(rank == world_size - 1)).to(device)

    micro_inputs = None
    if rank == 0:
        g = torch.Generator().manual_seed(2024)
        micro_inputs = [torch.randn(MICRO_BS, DIM, generator=g).to(device) for _ in range(NUM_MICRO)]

    num_warmup = min(world_size - 1 - rank, NUM_MICRO)
    rank_print(rank, f"stage {rank} | warmup={num_warmup} | async: weights step per micro-batch (no flush)")

    timeline = Timeline()
    pipedream_schedule(rank, world_size, stage, micro_inputs, device, timeline)

    rank0_print(rank, render_gantt(timeline, world_size))
    rank0_print(rank, "PipeDream: no global flush, weights stashed per micro-batch -> full pipe, consistent grads.")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
