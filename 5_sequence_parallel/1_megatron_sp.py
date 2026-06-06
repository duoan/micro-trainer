"""Module 5 (Sequence Parallelism) - Baseline: Megatron SP -- the All-Gather / Reduce-Scatter conjugate pair.

Principle in one line (Megatron-LM Sequence Parallelism):
    Pure Tensor Parallel (Module 2) leaves the regions OUTSIDE the matmuls (LayerNorm,
    Dropout, residual) with FULLY REPLICATED activations on every TP rank -- wasting
    activation memory. SP shards those regions along the SEQUENCE dimension too: each
    rank holds only [S/n, D] of the activation.

    The transition between the two worlds is a pair of conjugate operators that replace
    pure TP's single All-Reduce, keeping the total communication volume identical:
        g     (enter TP region) : forward = All-Gather along sequence; backward = Reduce-Scatter
        g-bar (exit  TP region) : forward = Reduce-Scatter along sequence; backward = All-Gather

    This demo runs the canonical Megatron block: sequence-sharded input -> [g] All-Gather
    -> column-parallel Linear -> ReLU -> row-parallel Linear (partial sums) -> [g-bar]
    Reduce-Scatter -> sequence-sharded output. The Reduce-Scatter SUM is correct because
    it sums the row-parallel partial outputs while scattering them along the sequence.

Your battle zone (two TODOs):
    - `enter_tp_region` (g):  All-Gather the [S/n, D] shards into the full [S, D].
    - `exit_tp_region`  (g-bar): Reduce-Scatter the [S, D] partial sums into [S/n, D] shards.

Run: python 5_sequence_parallel/1_megatron_sp.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: F401  (you will write all_gather / reduce_scatter in the TODOs)

from env_setup import launch_teaching_cluster, rank0_print, rank_print

WORLD_SIZE = 4
SEQ, DIM, FFN = 16, 8, 32  # SEQ divisible by WORLD_SIZE; FFN divisible by WORLD_SIZE
SHARD = SEQ // WORLD_SIZE


def build_full_inputs_and_weights() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All ranks build the same full X, A (D->FFN), B (FFN->D) for cross-checking."""
    g = torch.Generator().manual_seed(11)
    x = torch.randn(SEQ, DIM, generator=g)
    a = torch.randn(DIM, FFN, generator=g)
    b = torch.randn(FFN, DIM, generator=g)
    return x, a, b


def enter_tp_region(x_local: torch.Tensor, world_size: int, device: torch.device) -> torch.Tensor:
    """g operator: All-Gather sequence shards [S/n, D] -> full [S, D].

    ============================ YOUR BATTLE ZONE 1 ==========================
    Steps:
        1. x_full = empty tensor of shape [SEQ, DIM] on x_local.device.
        2. dist.all_gather_into_tensor(x_full, x_local.contiguous())
           (this concatenates each rank's shard along dim 0, in rank order).
        3. return x_full.
    ==========================================================================
    """
    # TODO(you): all_gather_into_tensor to gather sequence shards into the full activation
    raise NotImplementedError("TODO: enter_tp_region (g) -- hand-write All-Gather along the sequence dim")


def exit_tp_region(z_partial: torch.Tensor, world_size: int, device: torch.device) -> torch.Tensor:
    """g-bar operator: Reduce-Scatter [S, D] partial sums -> sequence-sharded [S/n, D].

    ============================ YOUR BATTLE ZONE 2 ==========================
    Steps:
        1. z_local = empty tensor of shape [SHARD, DIM] on z_partial.device.
        2. dist.reduce_scatter_tensor(z_local, z_partial.contiguous(),
                                      op=dist.ReduceOp.SUM)
           (this sums z_partial across all ranks, then scatters chunk r to rank r).
        3. return z_local.
    ==========================================================================
    """
    # TODO(you): reduce_scatter_tensor to sum partials and scatter along the sequence dim
    raise NotImplementedError("TODO: exit_tp_region (g-bar) -- hand-write Reduce-Scatter along the sequence dim")


def run(rank: int, world_size: int, device: torch.device) -> None:
    assert SEQ % world_size == 0 and FFN % world_size == 0

    x_full, a_full, b_full = build_full_inputs_and_weights()

    # SP region: each rank holds only its sequence shard of the input.
    x_local = x_full[rank * SHARD : (rank + 1) * SHARD].to(device)
    # Column-parallel shard of A (split along FFN), row-parallel shard of B (split along FFN).
    per = FFN // world_size
    a_shard = a_full[:, rank * per : (rank + 1) * per].to(device)
    b_shard = b_full[rank * per : (rank + 1) * per, :].to(device)

    rank_print(rank, f"sequence shard x_local = {tuple(x_local.shape)} (full seq = {SEQ})")

    # [g] enter TP region: gather full activation along the sequence.
    x = enter_tp_region(x_local, world_size, device)            # [SEQ, DIM]
    # Column-parallel Linear + ReLU (Module 2 knowledge), then row-parallel Linear -> partial sum.
    h = torch.relu(x @ a_shard)                                 # [SEQ, FFN/n]
    z_partial = h @ b_shard                                     # [SEQ, DIM] partial sum
    # [g-bar] exit TP region: reduce-scatter partials back to a sequence shard.
    z_local = exit_tp_region(z_partial, world_size, device)     # [SHARD, DIM]

    # Correctness self-check: gather all shards and compare to the single-machine reference.
    gathered = torch.empty(SEQ, DIM, device=device)
    dist.all_gather_into_tensor(gathered, z_local.contiguous())
    ref = (torch.relu(x_full @ a_full) @ b_full).to(device)
    err = (gathered - ref).abs().max().item()
    rank0_print(rank, f"SP output (gathered) shape = {tuple(gathered.shape)} | max error vs single-machine = {err:.2e}")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
