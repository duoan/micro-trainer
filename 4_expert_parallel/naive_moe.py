"""Module 4 (Expert Parallelism) - Baseline: Serial MoE (All-to-All dumb waiting, stuck in network latency).

Principle in one line:
    MoE (Mixture of Experts) = each rank hosts one/few "experts" (an FFN). Each token
    is routed to the rank hosting its target expert. The flow is the classic three steps:
        1. Dispatch: use All-to-All to send local tokens to the rank of their target expert.
        2. Expert: the local expert runs its forward on the received tokens.
        3. Combine: another All-to-All to send results BACK to their origin rank.

    The serial (naive) pain point: the Dispatch All-to-All is SYNCHRONOUS and BLOCKING.
    During the cross-machine latency, the powerful compute units can only sit and stare.
    Expert compute and communication are fully serialized; total time ~= comm + compute,
    with the latency hidden away not at all.

    Note the makespan (total time) after running. The next lesson, lightning_moe, uses
    Tile overlap to crush it.

Your battle zone:
    - `naive_moe_forward`: dutifully dispatch -> expert -> combine, with synchronous All-to-All throughout.

Run: python 4_expert_parallel/naive_moe.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from topology_sim import (  # noqa: F401  (delay-injected synchronous All-to-All)
    DEFAULT_LINK,
    slow_all_to_all_single,
)

from env_setup import (
    COMM_DEVICE,  # noqa: F401  (remember to move comm tensors onto it)
    Timeline,
    launch_teaching_cluster,
    rank0_print,
    rank_print,
)

WORLD_SIZE = 4
TOKENS_PER_EXPERT = 32  # tokens this rank sends to EACH expert (balanced-routing assumption)
DIM = 64


class Expert(nn.Module):
    """The simplest expert = one FFN layer. Each rank hosts one."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def make_local_tokens(rank: int, world_size: int, device: torch.device) -> torch.Tensor:
    """Build this rank's input tokens, shape [world_size * TOKENS_PER_EXPERT, DIM].

    Layout convention: the e-th segment (TOKENS_PER_EXPERT rows each) is "tokens to
    send to expert e". This way All-to-All's equal-split semantics map exactly to
    "dispatch by target expert".
    """
    g = torch.Generator().manual_seed(7000 + rank)
    return torch.randn(world_size * TOKENS_PER_EXPERT, DIM, generator=g).to(device)


def naive_moe_forward(
    rank: int,
    world_size: int,
    expert: Expert,
    tokens: torch.Tensor,
    device: torch.device,
    timeline: Timeline,
) -> torch.Tensor:
    """Serial MoE forward: synchronous dispatch -> local expert compute -> synchronous combine.

    ============================ YOUR BATTLE ZONE ============================
    1) Dispatch (synchronous All-to-All):
         dispatched = empty_like(tokens) on COMM_DEVICE
         wrap with timeline.span(rank, "dispatch", kind="comm"):
             slow_all_to_all_single(dispatched, tokens, DEFAULT_LINK)
       Now `dispatched` holds "tokens from the whole world destined for this rank's expert".

    2) Expert (compute):
         wrap with timeline.span(rank, "expert", kind="compute"):
             out = expert(dispatched.to(device))

    3) Combine (synchronous All-to-All back):
         combined = empty_like(out) on COMM_DEVICE
         wrap with timeline.span(rank, "combine", kind="comm"):
             slow_all_to_all_single(combined, out, DEFAULT_LINK)
       return combined.to(device).

    Feel it: the three steps are strictly back-to-back with zero overlap; compute units
    idle entirely during the communication latency.
    ==========================================================================
    """
    # TODO(you): dispatch -> expert -> combine, fully synchronous and serial
    raise NotImplementedError("TODO: naive_moe_forward -- hand-write synchronous serial dispatch/expert/combine")


def run(rank: int, world_size: int, device: torch.device) -> None:
    torch.manual_seed(300 + rank)
    expert = Expert().to(device)
    tokens = make_local_tokens(rank, world_size, device)

    rank_print(rank, f"local tokens = {tuple(tokens.shape)} | simulated link latency = {DEFAULT_LINK.latency_s * 1000:.0f}ms/transfer")

    timeline = Timeline()
    _ = naive_moe_forward(rank, world_size, expert, tokens, device, timeline)

    makespan = timeline.makespan()
    rank0_print(rank, f"naive MoE makespan = {makespan * 1000:.1f} ms (comm and compute fully serialized)")
    rank0_print(rank, "Note this number down and compare with lightning_moe.py's ASCII bar.")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
