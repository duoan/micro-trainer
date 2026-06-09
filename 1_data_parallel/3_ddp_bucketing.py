"""Module 1 - Level 3: Data Parallel with bucketed, overlapped gradient sync.

Principle in one line:
    Level 2 overlapped communication by firing ONE async All-Reduce per parameter the
    instant its grad was ready. That hides latency, but a model with N parameter tensors
    still pays N collective launches every step. Real DDP does both ideas at once: it
    groups parameters into BUCKETS, and the moment every grad in a bucket is ready it
    fires a SINGLE async All-Reduce for the whole bucket. Fewer, larger collectives --
    and each one still overlaps with the backward that's computing earlier layers.

    This is exactly what torch DDP does with its ~25 MB buckets. We hand you the bucket
    assembly and the "is this bucket full yet?" bookkeeping; you write the part that
    actually reduces a full bucket asynchronously and scatters the averaged result back.

(On a single CPU box there's no wall-clock win to see -- the lesson is the MECHANISM:
buckets + async collectives + a wait barrier. The same code is what scales on GPUs.)

Run: python 1_data_parallel/3_ddp_bucketing.py
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
BUCKET_SIZE = 2  # parameter tensors per bucket (tiny, so TinyMLP splits into 2 buckets)


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
    """DDP wrapper that reduces gradients one BUCKET at a time, overlapped with backward.

    Builds on Level 2: instead of an async All-Reduce per parameter, params are grouped
    into buckets and a whole bucket is reduced (asynchronously) the moment its last grad
    lands. The per-step workflow is identical to the overlap demo:
        ddp.overlap_enabled = True
        loss.backward()          # hooks fill buckets; full buckets fire async reduces
        ddp.finish_overlap()     # wait for them all, average + scatter back
    """

    def __init__(self, module: nn.Module, world_size: int, bucket_size: int = BUCKET_SIZE) -> None:
        super().__init__()
        self.module = module
        self.world_size = world_size
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        # --- provided plumbing: bucket assembly + readiness tracking ---
        # Grads become ready in roughly REVERSE parameter order (last layer first), so we
        # bucket params in that order, `bucket_size` tensors at a time -- the bucket that
        # fills first is then the first one we can reduce.
        params = [p for p in self.module.parameters() if p.requires_grad]
        params.reverse()
        self._buckets: list[list[nn.Parameter]] = [
            params[i : i + bucket_size] for i in range(0, len(params), bucket_size)
        ]
        self._bucket_of = {p: bi for bi, b in enumerate(self._buckets) for p in b}
        self._pending = [len(b) for b in self._buckets]  # grads still missing per bucket
        self.overlap_enabled = False
        # finish_overlap consumes whatever _reduce_bucket records here.
        self._handles: list[tuple[dist.Work, torch.Tensor, list[nn.Parameter]]] = []
        for p in params:
            p.register_post_accumulate_grad_hook(self._hook)

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return self.module(*args, **kwargs)

    def _hook(self, param: nn.Parameter) -> None:
        """Provided: counts a bucket down and, once it's complete, hands it to you."""
        if not self.overlap_enabled:
            return
        bi = self._bucket_of[param]
        self._pending[bi] -= 1
        if self._pending[bi] == 0:  # every grad in this bucket is now ready
            self._reduce_bucket(self._buckets[bi])
            self._pending[bi] = len(self._buckets[bi])  # reset for the next step

    def _reduce_bucket(self, params: list[nn.Parameter]) -> None:
        """Fires once per bucket, the moment ALL of its grads are ready.

        ========================= YOUR BATTLE ZONE (a) =========================
        Pack this bucket's grads into ONE contiguous buffer, launch a single non-blocking
        All-Reduce(SUM) on it, and stash everything `finish_overlap` will need to unpack
        it later (the work handle, the buffer, and which params it came from). Don't
        average yet -- the async reduce only sums; the divide happens after it lands.
        ========================================================================
        """
        # TODO(you): flatten this bucket's grads, fire one async All-Reduce, record it.
        raise NotImplementedError("TODO(a): reduce a full bucket asynchronously in _reduce_bucket")

    def finish_overlap(self) -> None:
        """Call AFTER backward(): drain every in-flight bucket reduce.

        ========================= YOUR BATTLE ZONE (b) =========================
        For each recorded bucket: wait for its reduce to land, turn the summed buffer into
        a mean (÷ world_size), then unpack each averaged slice back into its param's .grad
        with the original shape. Reset the list afterwards. (Mind the per-param offsets.)
        ========================================================================
        """
        # TODO(you): wait on each bucket handle, average, scatter slices back, then clear.
        raise NotImplementedError("TODO(b): wait + average + scatter the buckets in finish_overlap")


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
    rank_print(rank, f"local shard = {tuple(x_local.shape)} | {len(ddp._buckets)} buckets, each reduced async when full")

    # Trusted reference (no collective, hooks dormant): per-shard average == full-batch grad.
    def snapshot() -> list[torch.Tensor]:
        return [p.grad.clone() for p in ddp.module.parameters() if p.grad is not None]

    ddp.zero_grad()
    loss_fn(ddp(x_full), y_full).backward()
    ref = snapshot()

    # Measured pass: enable overlap, backward fires the per-bucket async reduces, then finish.
    ddp.overlap_enabled = True
    ddp.zero_grad()
    loss_fn(ddp(x_local), y_local).backward()
    ddp.finish_overlap()
    ddp.overlap_enabled = False
    err = max((g - r).abs().max().item() for g, r in zip(snapshot(), ref, strict=True))
    rank0_print(rank, f"bucketed-overlap sync | max err vs full-batch grad = {err:.2e}")

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

    rank0_print(rank, "Bucketed-overlap DDP done: one async reduce per bucket -- the real DDP combo.")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
