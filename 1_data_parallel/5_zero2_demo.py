"""Module 1 - Evolution 2: ZeRO-2 as a single ShardedDDP wrapper.

Principle in one line:
    ZeRO-1 (demo 4) was an OPTIMIZER: it sharded the optimizer state but averaged grads
    with a plain `all_reduce`, so every rank still held the FULL gradient. ZeRO-2 goes
    one step further -- and we fold the whole thing into ONE wrapper.

    `MicroShardedDistributedDataParallel` wraps the model and owns everything ZeRO-2
    needs, internally:
      * Optimizer-state sharding: it builds the base optimizer over ONLY the params this
        rank owns (so Adam's m/v live on one rank, exactly like ZeRO-1).
      * Gradient sharding: a backward hook reduces each grad to its OWNER during the
        backward pass (overlapped, like the DDP overlap demo), and `step()` frees the
        grads no rank owns -- so gradient memory also drops to ~1/world_size.

    One object handles grads + optimizer state. (FairScale splits this into OSS +
    ShardedDataParallel; we keep a single class so the whole ZeRO-2 story lives in one
    place.) Parameters are still replicated -- sharding those too is ZeRO-3 / FSDP.

Run: python 1_data_parallel/5_zero2_demo.py
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

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


class MicroShardedDistributedDataParallel(nn.Module):
    """A from-scratch ShardedDDP: one wrapper that shards BOTH gradients and optimizer state.

    On construction it partitions the parameters across ranks, builds the base optimizer
    over only this rank's owned params (optimizer-state sharding), and registers a
    backward hook on every parameter. The per-step workflow is DDP-like, with the
    optimizer folded in:

        ddp.zero_grad()
        loss = loss_fn(ddp(x), y)
        loss.backward()   # hooks reduce each grad toward its owner, asynchronously
        ddp.step()        # wait for the reduces, free non-owned grads, sharded step, broadcast
    """

    def __init__(
        self,
        module: nn.Module,
        world_size: int,
        rank: int,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        **opt_kwargs: Any,
    ) -> None:
        super().__init__()
        self.module = module
        self.world_size = world_size
        self.rank = rank

        # Make every rank start identical (precondition for any data-parallel scheme).
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        # Round-robin ownership by whole tensor: owner[i] = rank that owns param i.
        params = list(self.module.parameters())
        self.owner = [i % world_size for i in range(len(params))]
        self.mine = [i for i, o in enumerate(self.owner) if o == rank]
        self._owner_of = {p: self.owner[i] for i, p in enumerate(params)}

        # Optimizer-state sharding handled HERE: base optimizer over owned params only.
        owned_params = [params[i] for i in self.mine]
        self.opt = optimizer_class(owned_params, **opt_kwargs)

        # finish_reduce (called by step) consumes whatever the hook records here.
        self._handles: list[tuple[dist.Work, nn.Parameter, int]] = []
        for p in params:
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._reduce_hook)

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return self.module(*args, **kwargs)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.module.parameters():
            p.grad = None

    def _reduce_hook(self, param: nn.Parameter) -> None:
        """Fires the instant `param.grad` is ready during backward(). Reduce it to its OWNER.

        ========================= YOUR BATTLE ZONE (a) =========================
        Only owner_of[param] needs this grad (it does the update). Turn the local grad
        into its share of the mean (scale by 1/world_size), then launch a non-blocking
        reduce(SUM) whose single destination is that owner, and stash what `step` will
        need. Which collective sums a tensor onto ONE destination rank (not all of them)?
        ========================================================================
        """
        # TODO(you): dst = self._owner_of[param]; scale grad by 1/world_size; launch an
        #            async reduce(SUM) to dst; record the handle (+ param + dst) for finish_reduce.
        raise NotImplementedError("TODO(a): reduce each grad toward its owner in _reduce_hook")

    def finish_reduce(self) -> None:
        """Called by `step` after backward(): drain the in-flight reduces, then shard grads.

        ========================= YOUR BATTLE ZONE (b) =========================
        Wait for every reduce to land. Then comes the actual memory win: this rank only
        keeps the grads it OWNS, so release every grad whose owner isn't this rank. Reset
        the list for the next step.
        ========================================================================
        """
        # TODO(you): wait on each handle; free grads this rank does not own; clear the list.
        raise NotImplementedError("TODO(b): wait then free non-owned grads in finish_reduce")

    @torch.no_grad()
    def _sync_params(self) -> None:
        """Provided: each updated param lives on its owner; broadcast it to everyone."""
        for i, p in enumerate(self.module.parameters()):
            dist.broadcast(p.data, src=self.owner[i])

    def step(self) -> None:
        self.finish_reduce()  # battle zone b: wait for reduces, free non-owned grads
        self.opt.step()       # sharded optimizer-state update on owned params
        self._sync_params()   # broadcast updated params so every rank holds full weights


def make_shard(rank: int, world_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(42)
    x = torch.randn(GLOBAL_BATCH, IN_DIM, generator=g)
    true_w = torch.randn(IN_DIM, OUT_DIM, generator=g)
    y = x @ true_w + 0.1 * torch.randn(GLOBAL_BATCH, OUT_DIM, generator=g)
    per = GLOBAL_BATCH // world_size
    sl = slice(rank * per, (rank + 1) * per)
    return x[sl].to(device), y[sl].to(device)


def run(rank: int, world_size: int, device: torch.device) -> None:
    torch.manual_seed(0)
    ddp = MicroShardedDistributedDataParallel(
        TinyMLP().to(device), world_size, rank, optimizer_class=torch.optim.Adam, lr=0.05
    )
    n_params = sum(1 for _ in ddp.module.parameters())
    rank_print(rank, f"params I own = {ddp.mine} (of {n_params} tensors) -> only these grads survive backward")

    loss_fn = nn.MSELoss()
    x, y = make_shard(rank, world_size, device)

    for step in range(STEPS):
        ddp.zero_grad()
        loss = loss_fn(ddp(x), y)
        loss.backward()  # hooks reduce each grad toward its owner
        ddp.step()       # finish reduce + sharded opt step + broadcast params
        rank_print(rank, f"step {step} | local_loss = {loss.item():.4f}")

    # ZeRO invariant: after broadcast, every rank must hold bit-identical weights.
    flat = torch.cat([p.data.flatten() for p in ddp.module.parameters()])
    ref = flat.clone()
    dist.broadcast(ref, src=0)
    drift = (flat - ref).abs().max().item()
    rank0_print(rank, f"ZeRO-2 done | cross-rank param drift = {drift:.2e} (0 == every rank in sync)")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
