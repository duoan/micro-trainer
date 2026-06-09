"""Module 1, Demo 2: Fully Sharded Data Parallel -- the ZeRO ladder (stages 1/2/3).

One `MicroFSDP(nn.Module)` wrapper that shards the model state across ranks in three
escalating stages. Every rank owns one contiguous chunk of the FLATTENED parameter
vector (rank r owns chunk r) and builds its optimizer over only that chunk, so the
optimizer state is sharded in all three stages. What grows is what ELSE gets sharded:

    zero1   shard optimizer STATE only. Grads are averaged with a full All-Reduce
            (every rank sees the whole gradient), each rank updates its own chunk,
            then an All-Gather rebuilds the full parameters.
    zero2   + shard GRADIENTS. Swap the All-Reduce for a Reduce-Scatter, so each rank
            only ever receives the averaged gradient for its OWN chunk.
    zero3   + shard PARAMETERS. Params live sharded at rest; All-Gather them just
            before forward and drop the full copy after the step. (PyTorch FSDP.)

The three "signature" collectives are your battle zones: All-Reduce (zero1),
Reduce-Scatter (zero2/zero3), and the parameter All-Gather (all stages).

    Memory: zero1 shards optimizer state; zero2 also grads; zero3 also params.

Run one stage:  python 1_data_parallel/2_fsdp.py zero1     (or zero2 / zero3)
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: E402  (communication written out in the open)
import torch.nn as nn
from _common import (  # noqa: E402, F401  (load_flat_into_model is used in the gather_params battle zone)
    STEPS,
    TinyMLP,
    cli,
    flatten_grads,
    flatten_params,
    load_flat_into_model,
    make_shard,
    pad_to_multiple,
)

from env_setup import rank0_print, rank_print  # noqa: E402

LEVELS = ["zero1", "zero2", "zero3"]


class MicroFSDP(nn.Module):
    """Hand-written FSDP/ZeRO. Each rank owns flat-chunk `rank` of the parameter vector
    and runs its optimizer over only that chunk (sharded optimizer state in every stage).

    Per step (orchestrated in `run`):
        [zero3] gather_params()      -> materialize full params from the shards
        forward + backward           -> full local gradients
        reduce_grads()               -> this rank's averaged gradient chunk -> owned.grad
        opt.step()                   -> update the owned chunk (sharded state)
        [zero1/zero2] gather_params()-> rebuild full params for the next forward
    """

    def __init__(
        self,
        module: nn.Module,
        world_size: int,
        rank: int,
        stage: str,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        **opt_kwargs: Any,
    ) -> None:
        super().__init__()
        self.module = module
        self.world_size = world_size
        self.rank = rank
        self.stage = stage
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

        self.total = sum(p.numel() for p in self.module.parameters())
        flat = pad_to_multiple(flatten_params(self.module), world_size)
        self.shard_size = flat.numel() // world_size
        # At rest each rank keeps ONLY its chunk; it is a leaf so the optimizer can step it.
        self.owned = flat.chunk(world_size)[rank].clone().requires_grad_(True)
        self.opt = optimizer_class([self.owned], **opt_kwargs)

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return self.module(*args, **kwargs)

    def zero_grad(self) -> None:
        for p in self.module.parameters():
            p.grad = None

    def gather_params(self) -> None:
        """Rebuild the full parameter vector from every rank's chunk and load it into the model.

        ===================== BATTLE ZONE: param All-Gather ======================
        `self.owned` is this rank's chunk (length shard_size). All-Gather every rank's
        chunk, concatenate into the full (padded) flat vector, trim the padding to
        `self.total`, and load it into the model with `load_flat_into_model`. This is the
        "materialize parameters from shards" step at the heart of FSDP.
        ==========================================================================
        """
        # TODO(you): all_gather owned chunks -> full flat -> load_flat_into_model(self.module, full[:total]).
        raise NotImplementedError("BATTLE ZONE [all_gather]: rebuild full params from the shards")

    def reduce_grads(self) -> None:
        """Set self.owned.grad to this rank's averaged gradient chunk (stage-dependent)."""
        full_grad = pad_to_multiple(flatten_grads(self.module), self.world_size)
        if self.stage == "zero1":
            self.owned.grad = self._all_reduce_grad(full_grad)
        else:
            self.owned.grad = self._reduce_scatter_grad(full_grad)

    def _all_reduce_grad(self, full_grad: torch.Tensor) -> torch.Tensor:
        """zero1: every rank averages the WHOLE gradient, then keeps its own chunk.

        ====================== BATTLE ZONE: grad All-Reduce ======================
        All-Reduce(SUM) the full gradient across ranks, divide by world_size, and return
        THIS rank's chunk -- `chunk(world_size)[self.rank]`. (Wasteful: every rank moves
        the whole gradient. zero2 fixes that with Reduce-Scatter.)
        ==========================================================================
        """
        # TODO(you): all_reduce(SUM) full_grad, /world_size, return chunk[self.rank].
        raise NotImplementedError("BATTLE ZONE [all_reduce]: average the full grad, keep my chunk")

    def _reduce_scatter_grad(self, full_grad: torch.Tensor) -> torch.Tensor:
        """zero2 / zero3: each rank receives ONLY the averaged gradient for its own chunk.

        ==================== BATTLE ZONE: grad Reduce-Scatter ====================
        Split full_grad into world_size contiguous chunks. Reduce-Scatter(SUM) them so
        rank r ends up with the summed r-th chunk, divide by world_size, and return it.
        (No rank ever holds a gradient slice it doesn't own -- gradient memory ~1/N.)
        ==========================================================================
        """
        # TODO(you): reduce_scatter the chunks into this rank's shard, /world_size, return it.
        raise NotImplementedError(
            "BATTLE ZONE [reduce_scatter]: scatter the averaged grad to its owner"
        )


def run(rank: int, world_size: int, device: torch.device, stage: str) -> None:
    torch.manual_seed(0)
    ddp = MicroFSDP(
        TinyMLP().to(device), world_size, rank, stage, optimizer_class=torch.optim.Adam, lr=0.05
    )
    rank_print(
        rank, f"[{stage}] my shard = {ddp.shard_size} elems (1/{world_size} of {ddp.total} params)"
    )

    loss_fn = nn.MSELoss()
    x, y = make_shard(rank, world_size, device)

    for step in range(STEPS):
        if stage == "zero3":
            ddp.gather_params()  # materialize full params from the shards (freed after the step)
        ddp.zero_grad()
        loss = loss_fn(ddp(x), y)
        loss.backward()
        ddp.reduce_grads()  # -> ddp.owned.grad (my averaged chunk)

        # self-check: my owned grad equals the matching chunk of the GLOBAL average grad.
        full_grad = pad_to_multiple(flatten_grads(ddp.module), world_size)
        dist.all_reduce(full_grad, op=dist.ReduceOp.SUM)
        ref = (full_grad / world_size).chunk(world_size)[rank]
        err = (ddp.owned.grad - ref).abs().max().item()

        ddp.opt.step()  # update the owned chunk (sharded optimizer state)
        if stage != "zero3":
            ddp.gather_params()  # rebuild full params on every rank for the next forward
        rank_print(
            rank, f"[{stage}] step {step} | local_loss={loss.item():.4f} | grad-chunk err={err:.2e}"
        )

    ddp.gather_params()  # ensure every rank holds the full, updated params
    flat = pad_to_multiple(flatten_params(ddp.module), world_size)
    ref = flat.clone()
    dist.broadcast(ref, src=0)
    drift = (flat - ref).abs().max().item()
    rank0_print(rank, f"[{stage}] FSDP done | cross-rank param drift = {drift:.2e} (0 == in sync)")


if __name__ == "__main__":
    cli(LEVELS, run)
