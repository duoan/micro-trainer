"""Module 2 - Evolution 3: 2D / 2.5D Tensor Parallel (SUMMA-style matmul on a grid).

Principle in one line (Colossal-AI 2D/2.5D, SUMMA):
    Megatron splits a weight along ONE dimension (1D TP). At large scale that 1D split
    makes each rank hold a fat slab and forces wide All-Gather/All-Reduce across all ranks.
    2D parallelism puts the world_size ranks on a sqrt(N) x sqrt(N) GRID and splits BOTH
    the rows and columns of every matrix into blocks. Y = X . W is then computed by SUMMA:

        rank (row, col) owns block X[row][col] and W[row][col].
        for k in 0..q-1:
            broadcast X[row][k] along the row group   (the rank at column k sends)
            broadcast W[k][col] along the column group (the rank at row k sends)
            acc += X[row][k] @ W[k][col]
        -> rank (row, col) ends holding output block Y[row][col].

    Each rank now holds only a 1/N block of every matrix (smaller than 1D's 1/sqrt(N) slab),
    and communication is confined to row/column groups of size sqrt(N) instead of all N ranks.
    (2.5D simply replicates this grid `depth` times to trade memory for even less comm.)

Your battle zone:
    - `summa_forward`: the broadcast-multiply-accumulate loop over the q grid steps.
      The row/column process groups are already built for you in `build_grid_groups`.

Run: python 2_tensor_parallel/3_summa_2d.py   (world_size must be a perfect square)
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: F401  (you will write dist.broadcast in the TODO)

from env_setup import COMM_DEVICE, launch_teaching_cluster, rank0_print, rank_print

WORLD_SIZE = 4  # must be a perfect square (2x2 grid here)
M, K, N = 8, 8, 8  # M, K, N must all be divisible by q = sqrt(world_size)


def build_full_matrices() -> tuple[torch.Tensor, torch.Tensor]:
    """All ranks build identical full X, W from a seed, for cross-checking against single-machine."""
    g = torch.Generator().manual_seed(7)
    x = torch.randn(M, K, generator=g)
    w = torch.randn(K, N, generator=g)
    return x, w


def block(mat: torch.Tensor, i: int, j: int, bi: int, bj: int) -> torch.Tensor:
    return mat[i * bi : (i + 1) * bi, j * bj : (j + 1) * bj].contiguous()


def build_grid_groups(rank: int, q: int):
    """Build the row and column process groups of the q x q grid (provided plumbing).

    rank -> (row, col) with row = rank // q, col = rank % q.
    NOTE: dist.new_group is collective, so every rank creates every row/col group.
    """
    row, col = rank // q, rank % q
    my_row_group = None
    for r in range(q):
        g = dist.new_group(ranks=[r * q + c for c in range(q)])
        if r == row:
            my_row_group = g
    my_col_group = None
    for c in range(q):
        g = dist.new_group(ranks=[r * q + c for r in range(q)])
        if c == col:
            my_col_group = g
    return my_row_group, my_col_group, row, col


def summa_forward(
    x_blk: torch.Tensor,
    w_blk: torch.Tensor,
    row: int,
    col: int,
    q: int,
    row_group,
    col_group,
    device: torch.device,
) -> torch.Tensor:
    """SUMMA: q rounds of (broadcast X along row) x (broadcast W along col) -> accumulate.

    x_blk = X[row][col] shape [M/q, K/q]; w_blk = W[row][col] shape [K/q, N/q].

    ============================ YOUR BATTLE ZONE ============================
    mb, kb = x_blk.shape ; _, nb = w_blk.shape
    acc = zeros [mb, nb] on COMM_DEVICE
    for k in range(q):
        # 1) X[row][k] lives on the rank at (row, k); its GLOBAL rank is row*q + k.
        #    Broadcast it to everyone in this row group.
        x_buf = x_blk.to(COMM_DEVICE).clone() if col == k else torch.empty(mb, kb, device=COMM_DEVICE)
        dist.broadcast(x_buf, src=row * q + k, group=row_group)

        # 2) W[k][col] lives on the rank at (k, col); its GLOBAL rank is k*q + col.
        #    Broadcast it to everyone in this column group.
        w_buf = w_blk.to(COMM_DEVICE).clone() if row == k else torch.empty(kb, nb, device=COMM_DEVICE)
        dist.broadcast(w_buf, src=k * q + col, group=col_group)

        # 3) accumulate the partial product
        acc += x_buf @ w_buf
    return acc.to(device)   # this rank now holds output block Y[row][col]
    ==========================================================================
    """
    # TODO(you): the q-round broadcast-multiply-accumulate SUMMA loop
    raise NotImplementedError("TODO: summa_forward -- hand-write the 2D SUMMA broadcast/accumulate loop")


def run(rank: int, world_size: int, device: torch.device) -> None:
    q = int(math.isqrt(world_size))
    assert q * q == world_size, "world_size must be a perfect square for the 2D grid"
    assert M % q == 0 and K % q == 0 and N % q == 0, "M, K, N must be divisible by q"

    full_x, full_w = build_full_matrices()
    bi_m, bi_k, bi_n = M // q, K // q, N // q

    row_group, col_group, row, col = build_grid_groups(rank, q)
    x_blk = block(full_x, row, col, bi_m, bi_k).to(device)
    w_blk = block(full_w, row, col, bi_k, bi_n).to(device)
    rank_print(rank, f"grid ({row},{col}) | X blk={tuple(x_blk.shape)} W blk={tuple(w_blk.shape)}")

    y_blk = summa_forward(x_blk, w_blk, row, col, q, row_group, col_group, device)

    # self-check: my output block must equal the matching block of single-machine X @ W
    ref = block(full_x @ full_w, row, col, bi_m, bi_n).to(COMM_DEVICE)
    err = (y_blk.to(COMM_DEVICE) - ref).abs().max().item()
    rank_print(rank, f"output block ({row},{col}) shape={tuple(y_blk.shape)} | max error vs single-machine={err:.2e}")
    rank0_print(rank, "2D SUMMA done: each rank held only a 1/N block; comm stayed within row/col groups.")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
