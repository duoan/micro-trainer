"""Module 1 - Evolution 1: hand-written ZeRO-1 (optimizer state sharding).

Principle in one line:
    Naive DDP's pain point: every rank keeps the FULL optimizer state (e.g. Adam's
    m/v, which is ~2x the parameter count). That memory is pure duplication.

    ZeRO-1's insight: not everyone needs the optimizer state. Partition the parameters
    across ranks and let EACH rank own/update only its own slice -- so the optimizer
    state for a parameter lives on exactly ONE rank instead of all N.

We package it as `MicroZeroOptimizer` -- a from-scratch stand-in for PyTorch's
`torch.distributed.optim.ZeroRedundancyOptimizer`. It wraps any base optimizer
(SGD, Adam, ...) but builds it over ONLY this rank's owned params, so the state is
genuinely sharded. Each `step()`:
    1. All-Reduce gradients into the global average (same as DDP).        [battle zone 1]
    2. Run the LOCAL optimizer -> updates only owned params (sharded state). [provided]
    3. All-Gather the updated params so every rank holds the full weights.   [battle zone 2]

    Optimizer-state memory saved = state * (1 - 1/world_size).

Run: python 1_data_parallel/4_zero1_demo.py
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Iterable
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


class MicroZeroOptimizer:
    """A from-scratch stand-in for torch.distributed.optim.ZeroRedundancyOptimizer (ZeRO-1).

    Wraps a base optimizer (e.g. torch.optim.Adam) but constructs it over ONLY the
    parameters this rank owns. That is the whole point: the optimizer state (Adam's
    m/v, SGD's momentum buffer, ...) is then allocated for ~1/world_size of the model
    on each rank instead of being replicated everywhere.

    `step()` averages the gradients, runs the local (sharded) optimizer, and then
    All-Gathers the freshly updated params so every rank ends the step holding the
    full, identical weight vector.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        world_size: int,
        rank: int,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        **opt_kwargs: Any,
    ) -> None:
        self.params: list[nn.Parameter] = list(params)
        self.world_size = world_size
        self.rank = rank

        # Round-robin ownership by whole tensor: owner[i] = rank that updates param i.
        # (Real ZeRO balances by element count; we shard whole tensors for clarity.)
        self.owner = [i % world_size for i in range(len(self.params))]
        self.mine = [i for i, o in enumerate(self.owner) if o == rank]

        # The crux: build the base optimizer over ONLY this rank's params, so its
        # state buffers exist for the owned shard alone -> sharded optimizer state.
        owned_params = [self.params[i] for i in self.mine]
        self.local_opt = optimizer_class(owned_params, **opt_kwargs)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def _average_grads(self) -> None:
        """Every rank needs the GLOBAL average gradient (same as DDP) before stepping.

        ============================ YOUR BATTLE ZONE 1 ==========================
        For each parameter's grad: turn each rank's local grad into the global mean.
        Which one collective sums a tensor across ranks, and what do you divide by?
        ==========================================================================
        """
        for p in self.params:
            if p.grad is None:
                continue
            # TODO(you): all_reduce(SUM) then divide by world_size.
            raise NotImplementedError("TODO(1): average each grad across ranks in _average_grads")

    @torch.no_grad()
    def _sync_params(self) -> None:
        """After the sharded step, only each owner holds its params' new values.

        ============================ YOUR BATTLE ZONE 2 ==========================
        Make every rank hold the full, up-to-date weights again: each parameter's new
        value lives on owner[i], so propagate it to everyone. (A per-param broadcast
        from src=owner[i] is the simplest All-Gather; all_gather + concat also works.)
        ==========================================================================
        """
        # TODO(you): send each owner's updated param to all ranks.
        raise NotImplementedError("TODO(2): all-gather/broadcast the updated params in _sync_params")

    def step(self) -> None:
        self._average_grads()  # battle zone 1: global grad average
        self.local_opt.step()  # provided: updates ONLY owned params (state is sharded)
        self._sync_params()    # battle zone 2: reconstruct the full param vector


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
    model = TinyMLP().to(device)
    # Broadcast initial weights so every rank starts identical (precondition for ZeRO).
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    opt = MicroZeroOptimizer(model.parameters(), world_size, rank, optimizer_class=torch.optim.Adam, lr=0.05)
    rank_print(rank, f"params I own = {opt.mine} (of {len(opt.params)} tensors) -> Adam state only for these")

    loss_fn = nn.MSELoss()
    x, y = make_shard(rank, world_size, device)

    for step in range(STEPS):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        rank_print(rank, f"step {step} | local_loss = {loss.item():.4f}")

    # ZeRO invariant: after All-Gather, every rank must hold bit-identical weights.
    flat = torch.cat([p.data.flatten() for p in model.parameters()])
    ref = flat.clone()
    dist.broadcast(ref, src=0)
    drift = (flat - ref).abs().max().item()
    rank0_print(rank, f"ZeRO-1 done | cross-rank param drift = {drift:.2e} (0 == every rank in sync)")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
