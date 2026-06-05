"""Module 1 - Evolution 2: hand-written ZeRO-3 / FSDP (Fully Sharded Data Parallel).

Principle in one line:
    ZeRO-1 sharded only the optimizer state; the full PARAMETERS still live on every
    rank. ZeRO-3 (a.k.a. FSDP) goes all the way: flatten every parameter into one big
    vector, cut it into world_size equal shards, and let each rank store ONLY its shard
    at rest. Parameters are materialized on demand:

        1. Just before compute: All-Gather the shards back into the full flat parameter
           vector, scatter it into the model, run forward/backward.
        2. Right after: drop the full copy again (re-shard), so peak parameter memory is
           ~1/world_size of the model (plus a transient full copy during the gather).
        3. Gradients are NOT All-Reduced; they are Reduce-Scattered, so each rank ends up
           holding only the averaged gradient for ITS OWN shard -- which is all it needs
           to update its shard.

    Memory: parameters + grads + optimizer state ALL drop by ~1/world_size. This is the
    strategy behind PyTorch FSDP and DeepSpeed ZeRO-3, the default for training big models.

Your battle zone (two TODOs below):
    - `all_gather_full_params`: All-Gather the per-rank shards into the full flat vector.
    - `reduce_scatter_grad`: Reduce-Scatter the full gradient so each rank keeps its shard.

Run: python 1_data_parallel/fsdp_demo.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: F401  (you will write dist.* in the TODOs)
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
    g = torch.Generator().manual_seed(42)
    x = torch.randn(GLOBAL_BATCH, IN_DIM, generator=g)
    true_w = torch.randn(IN_DIM, OUT_DIM, generator=g)
    y = x @ true_w + 0.1 * torch.randn(GLOBAL_BATCH, OUT_DIM, generator=g)
    per = GLOBAL_BATCH // world_size
    sl = slice(rank * per, (rank + 1) * per)
    return x[sl].to(device), y[sl].to(device)


# --------------------------------------------------------------------------- #
# Provided plumbing: flatten / unflatten the model <-> one flat vector.
# (Real FSDP does exactly this "flatten + pad to a multiple of world_size".)
# --------------------------------------------------------------------------- #
def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def load_flat_into_model(model: nn.Module, flat: torch.Tensor) -> None:
    off = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[off : off + n].view_as(p))
        off += n


def flatten_grads(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()])


def pad_to_multiple(flat: torch.Tensor, world_size: int) -> torch.Tensor:
    pad = (-flat.numel()) % world_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    return flat


def all_gather_full_params(local_shard: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-Gather every rank's shard back into the full (padded) flat parameter vector.

    ============================ YOUR BATTLE ZONE 1 ==========================
    local_shard is this rank's slice (length shard_size), already on any device.
    Steps:
        1. Build a list `gathered` of world_size empty tensors shaped like local_shard,
           on COMM_DEVICE.
        2. dist.all_gather(gathered, local_shard.to(COMM_DEVICE))
        3. torch.cat(gathered) -> the full padded flat vector; return it.
    This is the "materialize parameters just before compute" step of FSDP.
    ==========================================================================
    """
    # TODO(you): all_gather the shards and concatenate into the full flat vector
    raise NotImplementedError("TODO: all_gather_full_params -- hand-write the parameter All-Gather")


def reduce_scatter_grad(full_grad_flat: torch.Tensor, world_size: int) -> torch.Tensor:
    """Reduce-Scatter the full gradient so each rank receives only its shard's average.

    ============================ YOUR BATTLE ZONE 2 ==========================
    full_grad_flat is this rank's full (padded) gradient vector. With Reduce-Scatter,
    rank r should end up with SUM over ranks of the r-th chunk, divided by world_size.
    Steps:
        1. shard_size = full_grad_flat.numel() // world_size
        2. input_list = list(full_grad_flat.to(COMM_DEVICE).chunk(world_size))
           (a list of world_size contiguous chunks)
        3. out = empty tensor of shape [shard_size] on COMM_DEVICE
        4. dist.reduce_scatter(out, [c.contiguous() for c in input_list], op=dist.ReduceOp.SUM)
        5. return out / world_size   (the averaged gradient for THIS rank's shard)
    ==========================================================================
    """
    # TODO(you): reduce_scatter the full grad into this rank's averaged shard
    raise NotImplementedError("TODO: reduce_scatter_grad -- hand-write the gradient Reduce-Scatter")


def run(rank: int, world_size: int, device: torch.device) -> None:
    torch.manual_seed(0)
    model = TinyMLP().to(device)

    total = sum(p.numel() for p in model.parameters())
    flat = pad_to_multiple(flatten_params(model).to(COMM_DEVICE), world_size)
    shard_size = flat.numel() // world_size
    # At rest each rank keeps ONLY its shard (this is the memory win).
    local_shard = flat.chunk(world_size)[rank].clone()
    rank_print(rank, f"params total={total} padded={flat.numel()} | my shard={shard_size} elems (1/{world_size})")

    loss_fn = nn.MSELoss()
    x, y = make_shard(rank, world_size, device)

    for step in range(STEPS):
        # 1) materialize full params from shards, scatter into the model
        full = all_gather_full_params(local_shard, world_size)[:total].to(device)
        load_flat_into_model(model, full)

        # 2) local forward/backward on this rank's data shard
        model.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()

        # 3) reduce-scatter grads -> my shard's averaged gradient
        full_grad = pad_to_multiple(flatten_grads(model).to(COMM_DEVICE), world_size)
        my_grad = reduce_scatter_grad(full_grad, world_size)

        # self-check: the shard from reduce_scatter must equal the matching slice of the
        # globally averaged gradient (computed here via a plain all_reduce for reference).
        ref = full_grad.clone()
        dist.all_reduce(ref, op=dist.ReduceOp.SUM)
        ref = (ref / world_size).chunk(world_size)[rank]
        err = (my_grad - ref).abs().max().item()

        # 4) update only my shard (SGD; real FSDP keeps only this shard's optimizer state)
        local_shard = local_shard - 0.05 * my_grad
        rank_print(rank, f"step {step} | local_loss={loss.item():.4f} | reduce_scatter err vs avg={err:.2e}")

    rank0_print(rank, "FSDP/ZeRO-3 done: params + grads + optimizer state all sharded 1/world_size.")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
