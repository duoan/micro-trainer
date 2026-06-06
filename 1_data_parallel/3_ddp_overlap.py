"""Module 1 - Level 3: Data Parallel with communication/computation overlap.

Principle in one line:
    Levels 1 and 2 wait for backward() to FULLY finish, then average gradients. But a
    parameter's gradient is ready the moment its layer's backward is done -- long before
    the earliest layers finish. Overlap exploits that: the instant a grad is ready, fire
    its All-Reduce asynchronously and let it fly WHILE backward keeps computing the rest.
    Communication then hides behind computation instead of running after it.

    We hook into autograd with `register_post_accumulate_grad_hook`: it fires per
    parameter as soon as `.grad` is populated. The hook launches a non-blocking
    All-Reduce; after backward() we wait on all of them and finish the averaging.
    This is the heart of how real DDP achieves near-linear scaling.

(On a single CPU box there's no wall-clock win to see -- the lesson is the MECHANISM:
hooks + async collectives + a wait barrier. The same code is what pays off on GPUs.)

Run: python 1_data_parallel/3_ddp_overlap.py
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
    """DDP wrapper that overlaps gradient All-Reduce with the backward pass.

    On init it broadcasts the weights AND registers a post-accumulate-grad hook on every
    parameter. The hook stays dormant unless `overlap_enabled` is True (so we can compute
    a clean reference without it firing). The workflow each step is:
        ddp.overlap_enabled = True
        loss.backward()          # hooks fire async All-Reduces as grads become ready
        ddp.finish_overlap()     # wait for them all, finish the averaging
    """

    def __init__(self, module: nn.Module, world_size: int) -> None:
        super().__init__()
        self.module = module
        self.world_size = world_size
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        # --- provided plumbing ---
        self.overlap_enabled = False
        self._handles: list[tuple[dist.Work, torch.Tensor]] = []
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._overlap_hook)

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return self.module(*args, **kwargs)

    def _overlap_hook(self, param: torch.Tensor) -> None:
        """Fires the instant `param.grad` is ready during backward(). Kick off THIS grad's
        All-Reduce now, asynchronously, and let it run while backward keeps going.

        ========================= YOUR BATTLE ZONE (a) =========================
        The `if not self.overlap_enabled: return` guard is provided. Start a non-blocking
        All-Reduce(SUM) on param.grad and stash whatever `finish_overlap` will need in
        `self._handles`. (Hint: dist.all_reduce has an async_op flag and returns a handle.)
        ========================================================================
        """
        if not self.overlap_enabled:
            return
        # TODO(you): launch an async All-Reduce on param.grad and remember its handle.
        # raise NotImplementedError("TODO(a): launch an async All-Reduce in _overlap_hook")

    def finish_overlap(self) -> None:
        """Call AFTER backward(): make sure every in-flight All-Reduce has landed, then
        finish the averaging (the async reduce only did the SUM).

        ========================= YOUR BATTLE ZONE (b) =========================
        Drain self._handles: wait on each in-flight reduce, finish averaging the grad it
        was reducing, and reset the list for the next step.
        ========================================================================
        """
        # TODO(you): wait on every handle, average, then clear the list.
        # raise NotImplementedError("TODO(b): wait on the async handles and average in finish_overlap")


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
    rank_print(rank, f"local shard = {tuple(x_local.shape)} | grads All-Reduce as they fall out of backward()")

    # Trusted reference (no collective, hooks dormant): per-shard average == full-batch grad.
    def snapshot() -> list[torch.Tensor]:
        return [p.grad.clone() for p in ddp.module.parameters() if p.grad is not None]

    ddp.zero_grad()
    loss_fn(ddp(x_full), y_full).backward()
    ref = snapshot()

    # Measured pass: enable overlap, backward fires the async reduces, then finish.
    ddp.overlap_enabled = True
    ddp.zero_grad()
    loss_fn(ddp(x_local), y_local).backward()
    ddp.finish_overlap()
    ddp.overlap_enabled = False
    err = max((g - r).abs().max().item() for g, r in zip(snapshot(), ref, strict=True))
    rank0_print(rank, f"overlapped sync | max err vs full-batch grad = {err:.2e}")

    opt = torch.optim.SGD(ddp.parameters(), lr=0.05)
    for step in range(STEPS):
        ddp.overlap_enabled = True
        ddp.zero_grad()
        loss = loss_fn(ddp(x_local), y_local)
        loss.backward()
        ddp.finish_overlap()
        ddp.overlap_enabled = False
        opt.step()
        rank_print(rank, f"step {step} | local_loss = {loss.item():.4f}")

    rank0_print(rank, "Overlapped DDP done: comm fired from backward hooks, same average as naive/bucketed.")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
