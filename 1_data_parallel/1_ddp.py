"""Module 1, Demo 1: Distributed Data Parallel, three ways.

One `MicroDDP(nn.Module)` wrapper -- a from-scratch stand-in for
`torch.nn.parallel.DistributedDataParallel` -- with three selectable levels that all
compute the SAME averaged gradient, just faster each time:

    naive      one All-Reduce PER parameter, after backward() finishes.
    overlap    fire each grad's All-Reduce from a backward hook, asynchronously, so
               communication overlaps with the rest of backward.
    bucketing  coalesce every grad into ONE flat buffer and All-Reduce it once,
               amortizing the fixed per-collective launch cost.

Data parallelism = each rank trains on a different slice of the batch, then averages
gradients so the step matches training on the full batch. Averaging = All-Reduce(SUM)
divided by world_size; the three levels differ only in HOW that reduction is issued.

Run one level:  python 1_data_parallel/1_ddp.py naive      (or overlap / bucketing)
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: E402  (communication written out in the open)
import torch.nn as nn
from _common import STEPS, TinyMLP, build_global_dataset, cli, shard_for  # noqa: E402

from env_setup import rank0_print, rank_print  # noqa: E402

LEVELS = ["naive", "overlap", "bucketing"]
BUCKET_SIZE = 2  # parameter tensors per bucket (TinyMLP has 4 -> 2 buckets)


class MicroDDP(nn.Module):
    """Hand-written DDP. `forward` delegates to the wrapped model; after backward you call
    `sync_grads()` to turn each rank's local gradients into the global average.

    On construction it broadcasts the weights from rank 0 so every rank starts identical.
    For the hook-based levels (overlap, bucketing) it also registers backward hooks; the
    hooks stay dormant unless `syncing` is True so we can compute a clean reference.
    """

    def __init__(self, module: nn.Module, world_size: int, level: str) -> None:
        super().__init__()
        self.module = module
        self.world_size = world_size
        self.level = level
        self.syncing = True
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        self._handles: list[dist.Work] = []
        if level == "overlap":
            for p in self.module.parameters():
                if p.requires_grad:
                    p.register_post_accumulate_grad_hook(self._overlap_hook)
        elif level == "bucketing":
            # Grads become ready last-layer-first, so bucket params in reverse order.
            params = [p for p in self.module.parameters() if p.requires_grad][::-1]
            self._buckets = [
                params[i : i + BUCKET_SIZE] for i in range(0, len(params), BUCKET_SIZE)
            ]
            self._bucket_of = {p: bi for bi, b in enumerate(self._buckets) for p in b}
            self._pending = [len(b) for b in self._buckets]
            self._bucket_handles: list[tuple[dist.Work, torch.Tensor, list[nn.Parameter]]] = []
            for p in params:
                p.register_post_accumulate_grad_hook(self._bucket_hook)

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return self.module(*args, **kwargs)

    # ---- level "naive" -------------------------------------------------- #
    def _sync_naive(self) -> None:
        """Average every parameter's gradient with one All-Reduce per parameter.

        ============================ BATTLE ZONE: naive ==========================
        Each rank holds gradients from its OWN shard only. For each p.grad, make every
        rank end up with the mean across ranks. Which collective sums a tensor across
        ranks, and what do you divide by to turn that sum into a mean?
        ==========================================================================
        """
        # TODO(you): per-parameter All-Reduce(SUM) + divide by world_size.
        raise NotImplementedError("BATTLE ZONE [naive]: average each grad across ranks")

    # ---- level "overlap" ------------------------------------------------ #
    def _overlap_hook(self, param: nn.Parameter) -> None:
        """Fires the instant param.grad is ready during backward().

        ========================== BATTLE ZONE: overlap ==========================
        Don't wait for backward to finish: scale this grad by 1/world_size, launch a
        NON-blocking All-Reduce(SUM) on it now, and stash the work handle in
        `self._handles` so `sync_grads` can wait on it later. (dist.all_reduce has an
        async_op flag and returns a handle.)
        ==========================================================================
        """
        if not self.syncing:
            return
        # TODO(you): scale grad by 1/world_size, async All-Reduce, append handle to self._handles.
        raise NotImplementedError("BATTLE ZONE [overlap]: async All-Reduce from the backward hook")

    # ---- level "bucketing" --------------------------------------------- #
    def _bucket_hook(self, param: nn.Parameter) -> None:
        """Provided: counts a bucket down and hands it to you once every grad is in."""
        if not self.syncing:
            return
        bi = self._bucket_of[param]
        self._pending[bi] -= 1
        if self._pending[bi] == 0:
            self._reduce_bucket(self._buckets[bi])
            self._pending[bi] = len(self._buckets[bi])

    def _reduce_bucket(self, params: list[nn.Parameter]) -> None:
        """Reduce a whole ready bucket with a single collective.

        ========================= BATTLE ZONE: bucketing =========================
        Pack this bucket's grads into ONE contiguous buffer, scale by 1/world_size, fire
        a single non-blocking All-Reduce(SUM) on the buffer, and record (handle, buffer,
        params) in `self._bucket_handles` so sync_grads can scatter the averaged result
        back into each grad (mind that grads have different shapes).
        ==========================================================================
        """
        # TODO(you): flatten the bucket, scale, async All-Reduce once, record for scatter-back.
        raise NotImplementedError("BATTLE ZONE [bucketing]: one fused All-Reduce per bucket")

    # ---- shared entry point --------------------------------------------- #
    def sync_grads(self) -> None:
        """Called after backward(): finish the gradient averaging for the active level."""
        if self.level == "naive":
            self._sync_naive()
        elif self.level == "overlap":
            for h in self._handles:
                h.wait()
            self._handles.clear()
        elif self.level == "bucketing":
            for handle, buf, params in self._bucket_handles:
                handle.wait()
                off = 0
                for p in params:
                    n = p.grad.numel()
                    p.grad.copy_(buf[off : off + n].view_as(p.grad))
                    off += n
            self._bucket_handles.clear()


def run(rank: int, world_size: int, device: torch.device, level: str) -> None:
    torch.manual_seed(0)
    x_full, y_full = build_global_dataset(device)
    x_local, y_local = shard_for(x_full, y_full, rank, world_size)
    loss_fn = nn.MSELoss()

    ddp = MicroDDP(TinyMLP().to(device), world_size, level)
    rank_print(
        rank, f"[{level}] local shard = {tuple(x_local.shape)} (global batch = {x_full.shape[0]})"
    )

    # Trusted reference (NO collective, hooks dormant): with equal shards and a mean loss,
    # the gradient averaged across shards equals the full-batch gradient.
    def snapshot() -> list[torch.Tensor]:
        return [p.grad.clone() for p in ddp.module.parameters() if p.grad is not None]

    ddp.syncing = False
    ddp.zero_grad()
    loss_fn(ddp(x_full), y_full).backward()
    ref = snapshot()
    ddp.syncing = True

    ddp.zero_grad()
    loss_fn(ddp(x_local), y_local).backward()
    ddp.sync_grads()
    err = max((g - r).abs().max().item() for g, r in zip(snapshot(), ref, strict=True))
    rank0_print(rank, f"[{level}] max err vs full-batch grad = {err:.2e}")

    opt = torch.optim.SGD(ddp.parameters(), lr=0.05)
    for step in range(STEPS):
        ddp.zero_grad()
        loss = loss_fn(ddp(x_local), y_local)
        loss.backward()
        ddp.sync_grads()
        opt.step()
        rank_print(rank, f"[{level}] step {step} | local_loss = {loss.item():.4f}")

    rank0_print(rank, f"[{level}] DDP done: averaged grads match the full-batch gradient.")


if __name__ == "__main__":
    cli(LEVELS, run)
