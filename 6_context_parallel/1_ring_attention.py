"""Module 6 (Context Parallelism) - Ring Attention with online softmax.

Principle in one line (Ring Attention / Context Parallelism):
    When the sequence is so long that even one attention matrix [S, S] won't fit, split
    the SEQUENCE across ranks: rank r holds Q_r, K_r, V_r for its own chunk of length L = S/n.
    But attention needs every query to see EVERY key/value. So we form a RING: each rank
    keeps its Q fixed, and rotates the (K, V) blocks around the ring (send to next, recv
    from prev). Over n steps, every K/V block visits every rank, and each rank accumulates
    the full attention for its local queries -- never materializing the full [S, S] matrix.

    The trick that makes incremental accumulation correct is ONLINE SOFTMAX (the same idea
    as FlashAttention): keep a running row-max `m`, running denominator `l`, and running
    weighted output `acc`, and rescale them as each new K/V block arrives.

This demo computes full (non-causal) attention via the ring, then checks it against a
single-machine reference. The ring exchange helper is provided; the loop + online-softmax
accumulation is yours.

Your battle zone:
    - `ring_attention`: loop n steps, ring-rotating (K, V) and accumulating with online softmax.

Run: python 6_context_parallel/1_ring_attention.py
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

from env_setup import COMM_DEVICE, launch_teaching_cluster, rank0_print, rank_print

WORLD_SIZE = 4
SEQ, DIM = 16, 8  # SEQ divisible by WORLD_SIZE
LOCAL = SEQ // WORLD_SIZE
SCALE = 1.0 / math.sqrt(DIM)


def build_full_qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All ranks build the same full Q, K, V for cross-checking against single-machine."""
    g = torch.Generator().manual_seed(13)
    q = torch.randn(SEQ, DIM, generator=g)
    k = torch.randn(SEQ, DIM, generator=g)
    v = torch.randn(SEQ, DIM, generator=g)
    return q, k, v


def ring_exchange(kv: torch.Tensor, rank: int, world_size: int, device: torch.device) -> torch.Tensor:
    """Rotate a (K|V packed) block one hop around the ring: send to rank+1, receive from rank-1.

    Uses non-blocking isend/irecv to avoid deadlock. kv is expected packed on the last
    dim (e.g. concat of K and V). Returns the block received from the previous rank.
    """
    nxt = (rank + 1) % world_size
    prv = (rank - 1) % world_size
    send_buf = kv.detach().to(COMM_DEVICE).contiguous()
    recv_buf = torch.empty_like(send_buf)
    reqs = [dist.isend(send_buf, dst=nxt), dist.irecv(recv_buf, src=prv)]
    for r in reqs:
        r.wait()
    return recv_buf.to(device)


def online_softmax_update(
    m: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    scores: torch.Tensor,
    v_blk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One online-softmax accumulation step (provided for you -- this is the FlashAttention core).

    Args:
        m:     running row-max, shape [L, 1]
        denom: running denominator (sum of exp), shape [L, 1]
        acc:   running weighted output, shape [L, DIM]
        scores: this block's attention logits Q @ K_blk^T * scale, shape [L, L]
        v_blk:  this block's values, shape [L, DIM]
    Returns the updated (m, denom, acc).
    """
    block_max = scores.max(dim=-1, keepdim=True).values          # [L, 1]
    m_new = torch.maximum(m, block_max)
    correction = torch.exp(m - m_new)                            # rescale old stats
    p = torch.exp(scores - m_new)                                # [L, L]
    denom_new = denom * correction + p.sum(dim=-1, keepdim=True)
    acc_new = acc * correction + p @ v_blk
    return m_new, denom_new, acc_new


def ring_attention(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    rank: int,
    world_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Ring Attention: keep Q fixed, rotate (K, V) around the ring, accumulate via online softmax.

    ============================ YOUR BATTLE ZONE ============================
    Setup running stats for the local queries:
        m     = full([LOCAL, 1], -inf)
        denom = zeros([LOCAL, 1])
        acc   = zeros([LOCAL, DIM])
    Start with the local (K, V) block: k_blk, v_blk = k_local, v_local.

    Loop `world_size` times (one per K/V block in the ring):
        1. scores = (q_local @ k_blk.transpose(0, 1)) * SCALE     # [LOCAL, LOCAL]
        2. m, denom, acc = online_softmax_update(m, denom, acc, scores, v_blk)
        3. if not the last step, rotate the block to the next rank:
              pack kv = torch.cat([k_blk, v_blk], dim=-1)
              kv = ring_exchange(kv, rank, world_size, device)
              k_blk, v_blk = kv[..., :DIM], kv[..., DIM:]

    After the loop: out = acc / denom   # normalize by the final denominator
    return out  # [LOCAL, DIM]

    Tip: this is non-causal full attention, so every block contributes; no masking needed.
    ==========================================================================
    """
    # TODO(you): the ring loop + online-softmax accumulation
    raise NotImplementedError("TODO: ring_attention -- hand-write the ring loop + online softmax accumulation")


def run(rank: int, world_size: int, device: torch.device) -> None:
    assert SEQ % world_size == 0

    q_full, k_full, v_full = build_full_qkv()
    sl = slice(rank * LOCAL, (rank + 1) * LOCAL)
    q_local = q_full[sl].to(device)
    k_local = k_full[sl].to(device)
    v_local = v_full[sl].to(device)

    rank_print(rank, f"local Q/K/V chunk = {tuple(q_local.shape)} (full seq = {SEQ})")

    out_local = ring_attention(q_local, k_local, v_local, rank, world_size, device)

    # Correctness self-check: gather all local outputs, compare to single-machine attention.
    gathered = torch.empty(SEQ, DIM, device=COMM_DEVICE)
    dist.all_gather_into_tensor(gathered, out_local.to(COMM_DEVICE).contiguous())
    ref = (torch.softmax((q_full @ k_full.transpose(0, 1)) * SCALE, dim=-1) @ v_full).to(COMM_DEVICE)
    err = (gathered - ref).abs().max().item()
    rank0_print(rank, f"Ring-Attention output shape = {tuple(gathered.shape)} | max error vs single-machine = {err:.2e}")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
