"""Module 1, Demo 3: Hybrid Sharded Data Parallel (HSDP) -- FSDP on a 2D mesh.

Pure FSDP shards parameters across ALL ranks, so the parameter All-Gather spans every
node and its (slow) inter-node links every step. HSDP arranges ranks on a 2D mesh:

    shard dimension   x   replica dimension
    (FSDP inside it)      (plain DDP across it)

Inside a SHARD GROUP (one node, fast NVLink) params/grads are FSDP-sharded. Across
REPLICA GROUPS (across nodes, slow) the model is replicated like DDP. So the expensive
parameter All-Gather stays node-local, and only a gradient All-Reduce crosses nodes.
This is `HYBRID_SHARD` in PyTorch FSDP.

The gradient sync becomes two steps on the mesh:
    1. Reduce-Scatter the gradient WITHIN the shard group  -> each rank owns one shard.
    2. All-Reduce that shard ACROSS the replica group       -> sum the replicas.
    3. Divide by world_size                                 -> global average shard.

Battle zone: `hsdp_sync_grad` (Reduce-Scatter in the shard group + All-Reduce in the
replica group). The 2D process groups are built for you in `build_mesh_groups`.

Run:  python 1_data_parallel/3_hsdp.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: E402  (communication written out in the open)
import torch.nn as nn
from _common import TinyMLP, flatten_grads, make_shard, pad_to_multiple  # noqa: E402

from env_setup import launch_teaching_cluster, rank0_print, rank_print  # noqa: E402

LEVELS = ["hsdp"]
WORLD_SIZE = 4
SHARD = 2  # ranks per shard group ("intra-node" FSDP width)
REPLICA = 2  # number of replicas ("inter-node" DDP width); SHARD * REPLICA == WORLD_SIZE
STEPS = 4


# --------------------------------------------------------------------------- #
# Provided plumbing: build the 2D mesh of process groups.
# Layout (replica-major): rank = replica_idx * SHARD + shard_idx.
#   shard group   = ranks sharing a replica_idx (intra-node FSDP)
#   replica group = ranks sharing a shard_idx   (inter-node DDP)
# NOTE: dist.new_group is collective -- EVERY rank must create EVERY subgroup,
# even ones it won't belong to. That is why we loop over all of them here.
# --------------------------------------------------------------------------- #
def build_mesh_groups(rank: int):
    replica_idx, shard_idx = rank // SHARD, rank % SHARD

    my_shard_group = None
    for rep in range(REPLICA):
        ranks = [rep * SHARD + s for s in range(SHARD)]
        g = dist.new_group(ranks=ranks)
        if rep == replica_idx:
            my_shard_group = g

    my_replica_group = None
    for s in range(SHARD):
        ranks = [rep * SHARD + s for rep in range(REPLICA)]
        g = dist.new_group(ranks=ranks)
        if s == shard_idx:
            my_replica_group = g

    return my_shard_group, my_replica_group, shard_idx


def hsdp_sync_grad(
    full_grad: torch.Tensor, shard_group, replica_group, shard_idx: int
) -> torch.Tensor:
    """Two-step mesh gradient sync: Reduce-Scatter in the shard group, All-Reduce in the replica group.

    ============================ YOUR BATTLE ZONE ============================
    full_grad is this rank's full (padded to a multiple of SHARD) gradient vector.

    Step 1 -- intra-node FSDP Reduce-Scatter over `shard_group` (size SHARD):
        chunks = list(full_grad.chunk(SHARD))   # one chunk per shard rank
        my = empty tensor shaped like chunks[0], on full_grad.device
        dist.reduce_scatter(my, [c.contiguous() for c in chunks],
                            op=dist.ReduceOp.SUM, group=shard_group)
        # now `my` = sum over the shard group of chunk[shard_idx]

    Step 2 -- inter-node DDP All-Reduce over `replica_group` (size REPLICA):
        dist.all_reduce(my, op=dist.ReduceOp.SUM, group=replica_group)
        # now `my` = sum over ALL ranks of chunk[shard_idx]

    Step 3 -- average: return my / WORLD_SIZE
    ==========================================================================
    """
    # TODO(you): reduce_scatter within shard_group, then all_reduce within replica_group, then /WORLD_SIZE
    raise NotImplementedError("BATTLE ZONE [hsdp]: reduce_scatter (shard) + all_reduce (replica)")


def run(rank: int, world_size: int, device: torch.device, level: str) -> None:
    assert SHARD * REPLICA == world_size, "SHARD * REPLICA must equal world_size"
    torch.manual_seed(0)
    model = TinyMLP().to(device)

    shard_group, replica_group, shard_idx = build_mesh_groups(rank)
    rank_print(rank, f"mesh coord: replica={rank // SHARD}, shard={shard_idx}")

    loss_fn = nn.MSELoss()
    x, y = make_shard(rank, world_size, device)

    for step in range(STEPS):
        model.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()

        full_grad = pad_to_multiple(flatten_grads(model), SHARD)
        my_grad = hsdp_sync_grad(full_grad, shard_group, replica_group, shard_idx)

        # self-check: equals the matching shard of the GLOBAL average gradient.
        ref = full_grad.clone()
        dist.all_reduce(ref, op=dist.ReduceOp.SUM)
        ref = (ref / world_size).chunk(SHARD)[shard_idx]
        err = (my_grad - ref).abs().max().item()
        rank_print(
            rank,
            f"step {step} | local_loss={loss.item():.4f} | mesh-sync err vs global avg={err:.2e}",
        )

    rank0_print(
        rank,
        "HSDP done: param All-Gather stays intra-node, only the grad All-Reduce crosses replicas.",
    )


if __name__ == "__main__":
    level = sys.argv[1] if len(sys.argv) > 1 else LEVELS[0]
    launch_teaching_cluster(WORLD_SIZE, run, level)
