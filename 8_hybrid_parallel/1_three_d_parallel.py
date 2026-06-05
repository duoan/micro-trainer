"""Module 8 - Capstone: 3D Parallelism (Data x Tensor x Pipeline combined).

Principle in one line (Megatron-DeepSpeed):
    Each axis you built in modules 1-3 attacks a different cost. Real GPT-scale training
    composes all three at once by laying the ranks on a 3D mesh:

        world_size = PP x DP x TP
        rank = pp_idx * (DP*TP) + dp_idx * TP + tp_idx

    and giving each axis its OWN set of process groups:
        - TP group  (fixed pp,dp; varies tp): All-Reduce partial matmuls WITHIN a layer.
        - PP group  (fixed dp,tp; varies pp): send/recv activations ACROSS pipeline stages.
        - DP group  (fixed pp,tp; varies dp): All-Reduce gradients across data replicas.

    A rank therefore belongs to three groups simultaneously and uses the right one for the
    right collective. This demo wires a 2-stage pipeline (PP=2), each stage a row-parallel
    linear over TP=2, replicated over DP=2 -- the smallest mesh that exercises all three axes.

Your battle zone (two TODOs):
    - `hybrid_forward`: TP All-Reduce within a stage + PP send/recv across stages.
    - `dp_average`: average a tensor across the DATA-parallel group only (not the whole world).
    The 3D mesh of process groups is already built for you in `build_3d_mesh`.

Run: python 8_hybrid_parallel/1_three_d_parallel.py   (world_size must equal PP*DP*TP)
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: F401  (you will write dist.* in the TODOs)

from env_setup import COMM_DEVICE, launch_teaching_cluster, rank0_print, rank_print

PP, DP, TP = 2, 2, 2  # pipeline x data x tensor; product must equal WORLD_SIZE
WORLD_SIZE = PP * DP * TP  # = 8
B, DIN, DH, DOUT = 4, 8, 8, 4  # DIN, DH must be divisible by TP


def build_full_weights() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    w1 = torch.randn(DIN, DH, generator=g)
    w2 = torch.randn(DH, DOUT, generator=g)
    return w1, w2


def build_input() -> torch.Tensor:
    return torch.randn(B, DIN, generator=torch.Generator().manual_seed(123))


def coords(rank: int) -> tuple[int, int, int]:
    pp_i = rank // (DP * TP)
    rem = rank % (DP * TP)
    return pp_i, rem // TP, rem % TP  # (pp_idx, dp_idx, tp_idx)


def build_3d_mesh(rank: int):
    """Build the TP / PP / DP process groups of the 3D mesh (provided plumbing).

    NOTE: dist.new_group is collective -- every rank creates every group, then keeps the
    three it belongs to.
    """
    pp_i, dp_i, tp_i = coords(rank)
    tp_group = pp_group = dp_group = None
    for p in range(PP):
        for d in range(DP):
            ranks = [p * (DP * TP) + d * TP + t for t in range(TP)]
            g = dist.new_group(ranks=ranks)
            if p == pp_i and d == dp_i:
                tp_group = g
    for d in range(DP):
        for t in range(TP):
            ranks = [p * (DP * TP) + d * TP + t for p in range(PP)]
            g = dist.new_group(ranks=ranks)
            if d == dp_i and t == tp_i:
                pp_group = g
    for p in range(PP):
        for t in range(TP):
            ranks = [p * (DP * TP) + d * TP + t for d in range(DP)]
            g = dist.new_group(ranks=ranks)
            if p == pp_i and t == tp_i:
                dp_group = g
    return tp_group, pp_group, dp_group


def hybrid_forward(
    rank: int,
    x_full: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    tp_group,
    pp_group,
    device: torch.device,
) -> torch.Tensor | None:
    """Forward across the TP and PP axes. Returns y on stage-1 ranks, h on stage-0 ranks.

    pp_idx == 0 is stage 0 (W1 + relu); pp_idx == 1 is stage 1 (W2). tp_idx selects the shard.

    ============================ YOUR BATTLE ZONE 1 ==========================
    pp_i, dp_i, tp_i = coords(rank)
    per_in = DIN // TP

    if pp_i == 0:   # ---- pipeline stage 0: row-parallel W1 ----
        x_col   = x_full[:, tp_i*per_in:(tp_i+1)*per_in].to(device)          # [B, DIN/TP]
        w1_shard = w1[tp_i*per_in:(tp_i+1)*per_in, :].to(device)            # [DIN/TP, DH]
        partial = (x_col @ w1_shard).to(COMM_DEVICE)                         # [B, DH] partial sum
        dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=tp_group)       # TP All-Reduce
        h = torch.relu(partial)                                             # full [B, DH]
        # send h to my pipeline partner (same dp,tp) on stage 1, via the PP group
        dst = rank + (DP * TP)                                               # pp_idx 0 -> 1
        dist.send(h.contiguous(), dst=dst, group=pp_group)
        return h.to(device)

    else:           # ---- pipeline stage 1: row-parallel W2 ----
        per_h = DH // TP
        h = torch.empty(B, DH, device=COMM_DEVICE)
        src = rank - (DP * TP)
        dist.recv(h, src=src, group=pp_group)                               # recv activations
        h_col   = h[:, tp_i*per_h:(tp_i+1)*per_h].to(device)               # [B, DH/TP]
        w2_shard = w2[tp_i*per_h:(tp_i+1)*per_h, :].to(device)             # [DH/TP, DOUT]
        partial = (h_col @ w2_shard).to(COMM_DEVICE)                        # [B, DOUT] partial sum
        dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=tp_group)      # TP All-Reduce
        return partial.to(device)                                          # full y [B, DOUT]
    ==========================================================================
    """
    # TODO(you): TP all_reduce within each stage + PP send/recv between stage 0 and stage 1
    raise NotImplementedError("TODO: hybrid_forward -- TP All-Reduce within a stage + PP send/recv across stages")


def dp_average(t: torch.Tensor, dp_group) -> torch.Tensor:
    """Average a tensor across the DATA-parallel group only (the DP axis of the mesh).

    ============================ YOUR BATTLE ZONE 2 ==========================
    buf = t.to(COMM_DEVICE)
    dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=dp_group)
    return buf / DP        # only DP ranks participate, so divide by DP (not world_size)
    ==========================================================================
    """
    # TODO(you): all_reduce within dp_group and divide by DP
    raise NotImplementedError("TODO: dp_average -- All-Reduce within the data-parallel group only")


def run(rank: int, world_size: int, device: torch.device) -> None:
    assert world_size == PP * DP * TP, "world_size must equal PP*DP*TP"
    assert DIN % TP == 0 and DH % TP == 0, "DIN, DH must be divisible by TP"
    pp_i, dp_i, tp_i = coords(rank)
    rank_print(rank, f"mesh coord pp={pp_i} dp={dp_i} tp={tp_i}")

    w1, w2 = build_full_weights()
    x_full = build_input()
    tp_group, pp_group, dp_group = build_3d_mesh(rank)

    out = hybrid_forward(rank, x_full, w1, w2, tp_group, pp_group, device)

    # forward self-check: stage-1 ranks must reproduce the single-machine network output
    ref = torch.relu(x_full @ w1) @ w2
    if pp_i == 1:
        err = (out.to(COMM_DEVICE) - ref).abs().max().item()
        rank_print(rank, f"stage1 output {tuple(out.shape)} | max error vs single-machine={err:.2e}")
    else:
        h_ref = torch.relu(x_full @ w1)
        err = (out.to(COMM_DEVICE) - h_ref).abs().max().item()
        rank_print(rank, f"stage0 hidden {tuple(out.shape)} | max error vs single-machine={err:.2e}")

    # DP-axis check: a per-replica tensor that differs only by dp_idx must average correctly
    probe = torch.full((2,), float(dp_i), device=device)
    avg = dp_average(probe, dp_group)
    expected = sum(range(DP)) / DP
    rank_print(rank, f"dp_average(probe) = {avg.tolist()} (expected {expected})")

    rank0_print(rank, "3D parallelism done: TP All-Reduce (intra-layer) + PP send/recv (inter-stage) + DP All-Reduce (replicas).")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
