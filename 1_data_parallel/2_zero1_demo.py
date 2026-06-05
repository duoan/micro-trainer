"""Module 1 - Evolution 1: hand-written ZeRO-1 (optimizer state sharding + All-Gather).

Principle in one line:
    Naive DDP's pain point: every rank keeps the FULL optimizer state (e.g. Adam's
    m/v, which is 2x the parameter count). Memory is wasted on duplicated copies.

    ZeRO-1's insight: not everyone needs the optimizer state. Shard the parameters
    into world_size pieces and let EACH rank own/update only its own slice (holding
    only that slice's optimizer state).
        1. After backward, All-Reduce gradients into the global average (same as DDP).
        2. Each rank uses only its own slice of the gradient to update its own params.
        3. After the update, All-Gather the updated slices back into the full params
           and broadcast them to everyone.

    Memory saved = optimizer state * (1 - 1/world_size).

Your battle zone (two TODOs below):
    - `reduce_average_gradients`: the same gradient averaging as DDP (warm-up).
    - `step_and_all_gather`: update only the params this rank owns, then All-Gather back.

Run: python 1_data_parallel/2_zero1_demo.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: F401  (you will write dist.* in the TODOs)
import torch.nn as nn

# COMM_DEVICE is provided: move every cross-rank comm tensor onto it (gloo = cpu only)
from env_setup import COMM_DEVICE, launch_teaching_cluster, rank0_print, rank_print  # noqa: F401

WORLD_SIZE = 4
STEPS = 5
GLOBAL_BATCH = 64
IN_DIM, HIDDEN, OUT_DIM = 16, 32, 1


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, OUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_shard(rank: int, world_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(42)
    x = torch.randn(GLOBAL_BATCH, IN_DIM, generator=g)
    true_w = torch.randn(IN_DIM, OUT_DIM, generator=g)
    y = x @ true_w + 0.1 * torch.randn(GLOBAL_BATCH, OUT_DIM, generator=g)
    per = GLOBAL_BATCH // world_size
    sl = slice(rank * per, (rank + 1) * per)
    return x[sl].to(device), y[sl].to(device)


def assign_param_owner(num_params: int, world_size: int) -> list[int]:
    """Decide which rank owns the update of each parameter tensor (simple round-robin).

    Returns a list of length num_params, owner[i] = rank that owns parameter i.
    (Advanced: real ZeRO does a more balanced flatten-and-split by element count;
    here we shard by whole tensors for clarity.)
    """
    return [i % world_size for i in range(num_params)]


def reduce_average_gradients(model: nn.Module, world_size: int) -> None:
    """Global gradient averaging -- identical to ddp_demo; nail this first as a warm-up.

    ============================ YOUR BATTLE ZONE 1 ==========================
    For each p.grad: move to COMM_DEVICE -> all_reduce(SUM) -> /world_size -> write back.
    ==========================================================================
    """
    for p in model.parameters():
        if p.grad is None:
            continue
        # TODO(you): all_reduce to average gradients
        raise NotImplementedError("TODO: reduce_average_gradients -- hand-write gradient All-Reduce averaging")


def step_and_all_gather(
    model: nn.Module,
    owner: list[int],
    rank: int,
    world_size: int,
    lr: float,
) -> None:
    """The soul of ZeRO-1: each rank updates only its own params, then All-Gather back to full params.

    ============================ YOUR BATTLE ZONE 2 ==========================
    Iterate over parameters (use enumerate to get index i):
        A. If owner[i] == rank: this is "my parameter". Take one SGD step with the
           local (already averaged) gradient: p.data -= lr * p.grad.
           (Real ZeRO-1 holds Adam m/v state here; this demo uses SGD to get the
           skeleton running first.)
        B. Regardless of ownership, after the update everyone must see the latest
           value: dist.broadcast(param_cpu, src=owner[i]) sends the owner's updated
           parameter to all ranks.
           (broadcast is an equivalent simplification of all_gather; you can also
           literally use all_gather + concatenation to stay closer to the paper.)

    Remember: all communication tensors must be on COMM_DEVICE! Use owner[i] with index i.
    ==========================================================================
    """
    # TODO(you): use owner[i] to decide local update, then broadcast/all_gather latest params to all ranks
    raise NotImplementedError("TODO: step_and_all_gather -- hand-write sharded update + All-Gather/Broadcast back")


def run(rank: int, world_size: int, device: torch.device) -> None:
    torch.manual_seed(0)
    model = TinyMLP().to(device)

    params = list(model.parameters())
    owner = assign_param_owner(len(params), world_size)
    mine = [i for i, o in enumerate(owner) if o == rank]
    rank_print(rank, f"params I own = {mine} (of {len(params)} parameter tensors)")

    loss_fn = nn.MSELoss()
    x, y = make_shard(rank, world_size, device)

    for step in range(STEPS):
        model.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()

        reduce_average_gradients(model, world_size)
        step_and_all_gather(model, owner, rank, world_size, lr=0.05)

        rank_print(rank, f"step {step} | local_loss = {loss.item():.4f}")

    rank0_print(rank, "ZeRO-1 done: optimizer state is sharded, memory footprint drops with world_size.")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
