"""Module 1 - Level 1: Data Parallel, the simplest correct gradient sync.

Principle in one line:
    Data parallelism = each rank gets a DIFFERENT slice of the data, runs its own
    forward + backward, and ends up with its OWN local gradients. Because all ranks
    share the same parameters, every step must AVERAGE everyone's gradients before
    the optimizer step -- that is what makes it equivalent to "training on one huge
    batch". Averaging gradients = one All-Reduce(SUM) divided by world_size.

We package it as `MicroDDP(nn.Module)` -- a from-scratch stand-in for PyTorch's
`torch.nn.parallel.DistributedDataParallel`. Wrap any model: MicroDDP broadcasts the
initial weights so every rank starts identical, `forward` delegates to the wrapped
model, and after backward() you call `sync_grads()` to average gradients across ranks.

This is the baseline. Two follow-ups make the SAME averaging faster:
    2_ddp_overlap.py   -- fire each grad's All-Reduce from a backward hook (overlap comm).
    3_ddp_bucketing.py -- reduce whole BUCKETS of grads async at once (the real DDP combo).

Run: python 1_data_parallel/1_ddp_demo.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: E402  (communication written out in the open)
import torch.nn as nn

from env_setup import launch_teaching_cluster, rank0_print, rank_print

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


class MicroDDP(nn.Module):
    """A from-scratch stand-in for torch.nn.parallel.DistributedDataParallel.

    On construction it broadcasts the weights from rank 0 so every rank starts
    identical (a precondition for DDP correctness). `forward` delegates to the wrapped
    module, so `ddp(x)` and `ddp.parameters()` behave like a bare model. After a local
    backward() you call `sync_grads()` to turn each rank's local gradients into the
    global average.
    """

    def __init__(self, module: nn.Module, world_size: int) -> None:
        super().__init__()
        self.module = module
        self.world_size = world_size
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return self.module(*args, **kwargs)

    def sync_grads(self) -> None:
        """Average gradients across all ranks: the global mean must replace each local grad.

        ========================= YOUR BATTLE ZONE =========================
        Each rank currently holds gradients computed on its OWN data shard only.
        Make every rank end up with the average of all ranks' gradients, parameter
        by parameter. Think: which single collective sums a tensor across ranks, and
        what do you divide by to turn a sum into a mean? (Skip params whose grad is None.)
        ====================================================================
        """
        # TODO(you): hand-write the per-parameter gradient averaging.
        raise NotImplementedError("TODO: average each parameter's gradient across ranks in sync_grads")


def build_global_dataset(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the full global dataset (identical on every rank via a fixed seed)."""
    g = torch.Generator().manual_seed(42)
    x = torch.randn(GLOBAL_BATCH, IN_DIM, generator=g)
    true_w = torch.randn(IN_DIM, OUT_DIM, generator=g)
    y = x @ true_w + 0.1 * torch.randn(GLOBAL_BATCH, OUT_DIM, generator=g)
    return x.to(device), y.to(device)


def shard_for(x: torch.Tensor, y: torch.Tensor, rank: int, world_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's contiguous slice of the global batch."""
    per = GLOBAL_BATCH // world_size
    sl = slice(rank * per, (rank + 1) * per)
    return x[sl], y[sl]


def run(rank: int, world_size: int, device: torch.device) -> None:
    torch.manual_seed(0)  # same seed on all ranks; the broadcast in MicroDDP is the safety net
    x_full, y_full = build_global_dataset(device)
    x_local, y_local = shard_for(x_full, y_full, rank, world_size)
    loss_fn = nn.MSELoss()

    ddp = MicroDDP(TinyMLP().to(device), world_size)
    rank_print(rank, f"local data shard = {tuple(x_local.shape)} (global batch = {GLOBAL_BATCH})")

    # Trusted reference (NO collective): with equal shards and a mean-reduction loss, the
    # gradient averaged across shards equals the gradient on the FULL batch. Every rank
    # can compute it locally, so it is a clean ground truth for your sync.
    def snapshot() -> list[torch.Tensor]:
        return [p.grad.clone() for p in ddp.module.parameters() if p.grad is not None]

    ddp.zero_grad()
    loss_fn(ddp(x_full), y_full).backward()
    ref = snapshot()

    ddp.zero_grad()
    loss_fn(ddp(x_local), y_local).backward()  # each rank now holds only its shard's local grads
    ddp.sync_grads()
    err = max((g - r).abs().max().item() for g, r in zip(snapshot(), ref, strict=True))
    rank0_print(rank, f"sync_grads | max err vs full-batch grad = {err:.2e}")

    # A real training loop on top of your sync.
    opt = torch.optim.SGD(ddp.parameters(), lr=0.05)
    for step in range(STEPS):
        ddp.zero_grad()
        loss = loss_fn(ddp(x_local), y_local)
        loss.backward()
        ddp.sync_grads()
        opt.step()
        rank_print(rank, f"step {step} | local_loss = {loss.item():.4f}")

    rank0_print(rank, "Naive DDP done: averaged grads match the full-batch gradient. Next: overlap, then bucketing.")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
