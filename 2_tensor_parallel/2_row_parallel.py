"""Module 2 - Evolution 2: Row Parallel Linear (All-Reduce summation).

Principle in one line (Megatron-LM):
    Still Y = X . W, but this time "row split" = split W along the IN dimension
    into world_size pieces (stacked by rows):
        W = [W0; W1; ...; W_{n-1}], each W_i of shape [in/n, out].
    Correspondingly, the input X is split by column: X = [X0 | X1 | ...],
    each X_i of shape [batch, in/n].

    The magic: X . W = sum_i (X_i . W_i).
    Each rank computes a FULL-shape [batch, out] partial sum Y_i = X_i . W_i,
    and summing all Y_i gives the full output -- that is exactly All-Reduce(SUM).

    Compared to column parallel: column uses All-Gather concat, row uses
    All-Reduce sum. Megatron's classic trick chains "column -> row" so that the
    two intermediate communications cancel into a single one.

Your battle zone:
    - `forward`: compute the partial sum Y_i locally, then All-Reduce(SUM) into full Y.

Run: python 2_tensor_parallel/row_parallel.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: F401  (you will write dist.all_reduce in the TODO)

from env_setup import COMM_DEVICE, launch_teaching_cluster, rank0_print, rank_print

WORLD_SIZE = 4
BATCH, IN_DIM, OUT_DIM = 8, 16, 32  # IN_DIM must be divisible by WORLD_SIZE


def build_full_weight() -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return torch.randn(IN_DIM, OUT_DIM, generator=g)


def local_row_shard(full_w: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Extract this rank's row shard W_i, shape [IN_DIM/world_size, OUT_DIM]."""
    per = IN_DIM // world_size
    return full_w[rank * per : (rank + 1) * per, :].contiguous()


def local_input_shard(x_full: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Input X is split by column too: X_i shape [BATCH, IN_DIM/world_size], aligned with row shard W_i."""
    per = IN_DIM // world_size
    return x_full[:, rank * per : (rank + 1) * per].contiguous()


def row_parallel_forward(
    x_shard: torch.Tensor,
    w_shard: torch.Tensor,
    rank: int,
    world_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Row-parallel forward: compute the partial sum locally, then All-Reduce(SUM) into full Y.

    ============================ YOUR BATTLE ZONE ============================
    Steps:
        1. Local matmul: y_partial = x_shard @ w_shard   # shape [BATCH, OUT_DIM] (full shape on every rank)
        2. Move to COMM_DEVICE, dist.all_reduce(y_partial, op=dist.ReduceOp.SUM)
        3. Move back to device and return -- now every rank holds the same full Y.
    ==========================================================================
    """
    # TODO(you): local matmul + all_reduce(SUM)
    raise NotImplementedError("TODO: row_parallel_forward -- hand-write local partial sum + All-Reduce summation")


def run(rank: int, world_size: int, device: torch.device) -> None:
    assert IN_DIM % world_size == 0, "IN_DIM must be divisible by world_size"

    full_w = build_full_weight()
    w_shard = local_row_shard(full_w, rank, world_size).to(device)

    gx = torch.Generator().manual_seed(123)
    x_full = torch.randn(BATCH, IN_DIM, generator=gx)
    x_shard = local_input_shard(x_full, rank, world_size).to(device)

    rank_print(rank, f"holding row shard W_i = {tuple(w_shard.shape)} | input shard X_i = {tuple(x_shard.shape)}")

    y = row_parallel_forward(x_shard, w_shard, rank, world_size, device)

    ref = x_full @ full_w
    err = (y.to(COMM_DEVICE) - ref).abs().max().item()
    rank0_print(rank, f"Row-parallel output shape = {tuple(y.shape)} | max error vs single-machine = {err:.2e}")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
