"""Module 1 - Baseline 1: Naive DDP (hand-written gradient All-Reduce averaging).

Principle in one line:
    Data parallelism = each rank gets a DIFFERENT slice of the data, runs its own
    forward + backward, and ends up with its OWN local gradients. Because all ranks
    share the same model parameters, every step must AVERAGE everyone's gradients
    before updating -- that is what makes it equivalent to "training on one huge batch".

    Averaging gradients = one All-Reduce(SUM) divided by world_size.

Your battle zone (see the TODO below):
    In `synchronize_gradients`, hand-write dist.all_reduce to sum each parameter's
    .grad across ranks and then average it.

Run: python 1_data_parallel/ddp_demo.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: E402  (communication written out in the open)
import torch.nn as nn

from env_setup import COMM_DEVICE, launch_teaching_cluster, rank0_print, rank_print

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
    """Build a global dataset, then slice it for the current rank. A fixed seed
    guarantees every rank slices the very same global set."""
    g = torch.Generator().manual_seed(42)
    x = torch.randn(GLOBAL_BATCH, IN_DIM, generator=g)
    true_w = torch.randn(IN_DIM, OUT_DIM, generator=g)
    y = x @ true_w + 0.1 * torch.randn(GLOBAL_BATCH, OUT_DIM, generator=g)

    per = GLOBAL_BATCH // world_size
    sl = slice(rank * per, (rank + 1) * per)
    return x[sl].to(device), y[sl].to(device)


def broadcast_initial_params(model: nn.Module) -> None:
    """Make every rank start from identical parameters (broadcast from rank 0).
    This is a precondition for DDP correctness."""
    for p in model.parameters():
        cpu = p.detach().to(COMM_DEVICE)
        dist.broadcast(cpu, src=0)
        p.data.copy_(cpu.to(p.device))


def synchronize_gradients(model: nn.Module, world_size: int) -> None:
    """Average gradients across all ranks (All-Reduce SUM, then divide by world_size).

    ============================ YOUR BATTLE ZONE ============================
    What to do: iterate over model.parameters(), and for each p.grad:
        1. Move it to COMM_DEVICE (gloo only accepts cpu tensors!).
        2. dist.all_reduce(grad_cpu, op=dist.ReduceOp.SUM)
        3. Divide by world_size to get the averaged gradient.
        4. Write it back into p.grad (moved back to p's device).

    Hint: skip parameters with no gradient (p.grad is None).
    ==========================================================================
    """
    for p in model.parameters():
        if p.grad is None:
            continue
        # TODO(you): hand-write dist.all_reduce here to average gradients.
        # Delete the raise below and implement the 4 steps above.
        raise NotImplementedError("TODO: implement cross-rank gradient All-Reduce averaging in synchronize_gradients")


def run(rank: int, world_size: int, device: torch.device) -> None:
    torch.manual_seed(0)  # same seed on all ranks, with broadcast as a safety net
    model = TinyMLP().to(device)
    broadcast_initial_params(model)

    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    x, y = make_shard(rank, world_size, device)

    rank_print(rank, f"local data shard = {tuple(x.shape)} (global batch = {GLOBAL_BATCH})")

    for step in range(STEPS):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()  # at this point each rank only has its own shard's local gradients

        synchronize_gradients(model, world_size)  # turn local grads into the global average

        opt.step()
        rank_print(rank, f"step {step} | local_loss = {loss.item():.4f}")

    rank0_print(rank, "Naive DDP done: if all ranks' final params match, gradient sync is correct.")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
