"""Module 2 - Baseline 2: Column Parallel Linear (All-Gather concatenation).

Principle in one line (Megatron-LM):
    A linear layer Y = X . W, where W has shape [in, out].
    "Column split" = split W along the OUT dimension into world_size pieces:
        W = [W0 | W1 | ... | W_{n-1}].
    Each rank holds only W_i (shape [in, out/n]) and computes Y_i = X . W_i
    (shape [batch, out/n]).

    The full output Y = [Y0 | Y1 | ...] requires concatenating each rank's Y_i
    along the last dimension -- that is exactly what All-Gather does.

    Key point: input X is REPLICATED in full on every rank; the output is split
    by column and All-Gathered back together at the end.

Your battle zone:
    - `forward`: each rank computes Y_i with its local W_i, then All-Gather along
      dim=-1 into the full Y.

Run: python 2_tensor_parallel/1_column_parallel.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: F401  (you will write dist.all_gather in the TODO)

from env_setup import COMM_DEVICE, launch_teaching_cluster, rank0_print, rank_print

WORLD_SIZE = 4
BATCH, IN_DIM, OUT_DIM = 8, 16, 32  # OUT_DIM must be divisible by WORLD_SIZE


def build_full_weight() -> torch.Tensor:
    """All ranks build the same "full" weight W with one seed, for cross-checking against single-machine."""
    g = torch.Generator().manual_seed(7)
    return torch.randn(IN_DIM, OUT_DIM, generator=g)


def local_column_shard(full_w: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Extract this rank's column shard W_i, shape [IN_DIM, OUT_DIM/world_size]."""
    per = OUT_DIM // world_size
    return full_w[:, rank * per : (rank + 1) * per].contiguous()


def column_parallel_forward(
    x: torch.Tensor,
    w_shard: torch.Tensor,
    rank: int,
    world_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Column-parallel forward: compute Y_i locally, then All-Gather along the last dim into full Y.

    ============================ YOUR BATTLE ZONE ============================
    Steps:
        1. Local matmul: y_local = x @ w_shard      # shape [BATCH, OUT_DIM/world_size]
        2. Prepare a list `gathered` of length world_size, each element an empty
           tensor with the same shape as y_local (on COMM_DEVICE).
        3. dist.all_gather(gathered, y_local.to(COMM_DEVICE))
        4. torch.cat(gathered, dim=-1) for the full Y, move back to device and return.
    ==========================================================================
    """
    # TODO(you): local matmul + all_gather concatenation along dim=-1
    raise NotImplementedError("TODO: column_parallel_forward -- hand-write local matmul + All-Gather concat")


def run(rank: int, world_size: int, device: torch.device) -> None:
    assert OUT_DIM % world_size == 0, "OUT_DIM must be divisible by world_size"

    full_w = build_full_weight()
    w_shard = local_column_shard(full_w, rank, world_size).to(device)

    # Input X is identical and full on every rank
    gx = torch.Generator().manual_seed(123)
    x = torch.randn(BATCH, IN_DIM, generator=gx).to(device)

    rank_print(rank, f"holding column shard W_i = {tuple(w_shard.shape)}")

    y = column_parallel_forward(x, w_shard, rank, world_size, device)

    # Correctness self-check: compare against the single-machine full-W result
    ref = x.to(COMM_DEVICE) @ full_w
    err = (y.to(COMM_DEVICE) - ref).abs().max().item()
    rank0_print(rank, f"Column-parallel output shape = {tuple(y.shape)} | max error vs single-machine = {err:.2e}")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
