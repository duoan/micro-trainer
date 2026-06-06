"""Module 1 - Level 2: Data Parallel with gradient bucketing.

Principle in one line:
    Naive DDP (Level 1) does one All-Reduce PER parameter. Every collective has a fixed
    launch/latency cost, so a model with N parameter tensors pays that cost N times each
    step. Gradient bucketing fixes this: pack all the grads into ONE contiguous buffer,
    fire a SINGLE All-Reduce, then unpack the averaged result back into each `.grad`.

    Same math as Level 1, far fewer calls. Real DDP groups grads into ~25 MB "buckets"
    for exactly this reason (and to overlap each bucket with backward -- see Level 3).

We keep the `MicroDDP(nn.Module)` wrapper from Level 1; only `sync_grads` changes.

Run: python 1_data_parallel/2_ddp_bucketing.py
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
    """Same wrapper as Level 1 (broadcast on init, delegate forward); the gradient sync
    now coalesces everything into a single bucket."""

    def __init__(self, module: nn.Module, world_size: int) -> None:
        super().__init__()
        self.module = module
        self.world_size = world_size
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return self.module(*args, **kwargs)

    def sync_grads(self) -> None:
        """Average every gradient with a SINGLE All-Reduce instead of one per parameter.

        ========================= YOUR BATTLE ZONE =========================
        Flatten all the grads into one contiguous 1-D buffer, average that buffer with a
        single All-Reduce(SUM)+divide, then write each averaged slice back into the grad
        it came from. The crux is bookkeeping: grads have different shapes, so you must
        track where each one lives in the flat buffer to restore it with the right shape.
        (No torch private helpers -- reshape/cat/copy_ are all you need.)
        ====================================================================
        """
        # TODO(you): build one flat bucket, all_reduce it once, then scatter back.
        raise NotImplementedError("TODO: implement single-bucket gradient All-Reduce in sync_grads")


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
    torch.manual_seed(0)
    x_full, y_full = build_global_dataset(device)
    x_local, y_local = shard_for(x_full, y_full, rank, world_size)
    loss_fn = nn.MSELoss()

    ddp = MicroDDP(TinyMLP().to(device), world_size)
    n_params = sum(1 for _ in ddp.module.parameters())
    rank_print(rank, f"local shard = {tuple(x_local.shape)} | {n_params} param tensors -> 1 bucketed All-Reduce")

    # Trusted reference (no collective): per-shard average == full-batch gradient.
    def snapshot() -> list[torch.Tensor]:
        return [p.grad.clone() for p in ddp.module.parameters() if p.grad is not None]

    ddp.zero_grad()
    loss_fn(ddp(x_full), y_full).backward()
    ref = snapshot()

    ddp.zero_grad()
    loss_fn(ddp(x_local), y_local).backward()
    ddp.sync_grads()
    err = max((g - r).abs().max().item() for g, r in zip(snapshot(), ref, strict=True))
    rank0_print(rank, f"bucketed sync | max err vs full-batch grad = {err:.2e}")

    opt = torch.optim.SGD(ddp.parameters(), lr=0.05)
    for step in range(STEPS):
        ddp.zero_grad()
        loss = loss_fn(ddp(x_local), y_local)
        loss.backward()
        ddp.sync_grads()
        opt.step()
        rank_print(rank, f"step {step} | local_loss = {loss.item():.4f}")

    rank0_print(rank, "Bucketed DDP done: one fused All-Reduce per step, same average as the naive version.")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
