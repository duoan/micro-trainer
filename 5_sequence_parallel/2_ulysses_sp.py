"""Module 5 (Sequence Parallelism) - Evolution: DeepSpeed-Ulysses SP (All-to-All head/sequence swap).

Principle in one line (DeepSpeed-Ulysses):
    Megatron SP gathers the full activation along the sequence to enter the tensor-parallel
    region. Ulysses takes a different route for ATTENTION: keep activations sequence-sharded
    everywhere, and right around the attention op, use All-to-All to SWAP the partition --
    from "sharded along sequence, all heads" to "full sequence, sharded along heads".

    Each rank then computes ordinary full-sequence attention for its OWN subset of heads
    (no ring, no online softmax -- just a plain local attention), and a second All-to-All
    swaps back to sequence-sharded. Communication is two All-to-All ops whose volume is
    independent of sequence length per message; for many-head models this is very cheap and
    is the standard way to push context length in DeepSpeed.

    Layout (this demo sets num_heads H == world_size n, so each rank ends up with 1 head):
        q_local: [S/n, HID]   (HID = H * D, sequence-sharded, all heads)
          --[All-to-All seq->head]-->  [S, HID/n]   (full sequence, this rank's head group)
          --[local attention per head]-->  [S, HID/n]
          --[All-to-All head->seq]-->  [S/n, HID]   (back to sequence-sharded)

Your battle zone (two TODOs):
    - `all2all_seq_to_head`: All-to-All that turns [S/n, HID] into [S, HID/n].
    - `all2all_head_to_seq`: the inverse All-to-All, [S, HID/n] back into [S/n, HID].

Run: python 5_sequence_parallel/2_ulysses_sp.py
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: F401  (you will write all_to_all_single in the TODOs)

from env_setup import COMM_DEVICE, launch_teaching_cluster, rank0_print, rank_print

WORLD_SIZE = 4
SEQ, HEADS, HEAD_DIM = 16, 4, 8  # HEADS == WORLD_SIZE so each rank gets exactly 1 head after the swap
HID = HEADS * HEAD_DIM
SHARD = SEQ // WORLD_SIZE
HID_SHARD = HID // WORLD_SIZE
SCALE = 1.0 / math.sqrt(HEAD_DIM)


def build_full_qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All ranks build the same full Q, K, V (shape [SEQ, HID]) for cross-checking."""
    g = torch.Generator().manual_seed(13)
    q = torch.randn(SEQ, HID, generator=g)
    k = torch.randn(SEQ, HID, generator=g)
    v = torch.randn(SEQ, HID, generator=g)
    return q, k, v


def multihead_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Single-machine multi-head attention reference, output [SEQ, HID]."""
    out = torch.empty(SEQ, HID)
    for h in range(HEADS):
        sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        qh, kh, vh = q[:, sl], k[:, sl], v[:, sl]
        out[:, sl] = torch.softmax((qh @ kh.transpose(0, 1)) * SCALE, dim=-1) @ vh
    return out


def all2all_seq_to_head(x_local: torch.Tensor, world_size: int, device: torch.device) -> torch.Tensor:
    """Swap partition from [S/n, HID] (seq-sharded, all heads) to [S, HID/n] (full seq, head group).

    ============================ YOUR BATTLE ZONE 1 ==========================
    The reshape that arranges rows "grouped by destination rank" is given; you write the
    All-to-All. Steps:
        1. Group local rows by destination head-group:
              inp = x_local.reshape(SHARD, world_size, HID_SHARD).transpose(0, 1).reshape(world_size * SHARD, HID_SHARD)
           (chunk j of `inp` along dim 0 is the data destined for rank j.)
        2. out = empty([world_size * SHARD, HID_SHARD]) on COMM_DEVICE
           dist.all_to_all_single(out, inp.to(COMM_DEVICE))
           (now chunk k of `out` came from rank k = that rank's sequence chunk for OUR head group;
            stacking the n chunks along dim 0 yields the full sequence in order.)
        3. return out.reshape(SEQ, HID_SHARD).to(device)
    ==========================================================================
    """
    # TODO(you): reshape "by destination", all_to_all_single, then view as [SEQ, HID/n]
    raise NotImplementedError("TODO: all2all_seq_to_head -- hand-write the seq->head All-to-All swap")


def all2all_head_to_seq(o_head: torch.Tensor, world_size: int, device: torch.device) -> torch.Tensor:
    """Inverse swap: from [S, HID/n] (full seq, head group) back to [S/n, HID] (seq-sharded, all heads).

    ============================ YOUR BATTLE ZONE 2 ==========================
    Steps:
        1. Group by destination rank (each rank keeps its own sequence chunk):
              inp = o_head.reshape(world_size, SHARD, HID_SHARD).reshape(world_size * SHARD, HID_SHARD)
        2. out = empty([world_size * SHARD, HID_SHARD]) on COMM_DEVICE
           dist.all_to_all_single(out, inp.to(COMM_DEVICE))
        3. Reassemble all head groups for our local sequence chunk:
              return out.reshape(world_size, SHARD, HID_SHARD).transpose(0, 1).reshape(SHARD, HID).to(device)
    ==========================================================================
    """
    # TODO(you): reshape "by destination", all_to_all_single, then view as [S/n, HID]
    raise NotImplementedError("TODO: all2all_head_to_seq -- hand-write the head->seq All-to-All swap")


def local_attention(q_head: torch.Tensor, k_head: torch.Tensor, v_head: torch.Tensor) -> torch.Tensor:
    """Plain full-sequence attention for this rank's head group (HID/n columns, == 1 head here)."""
    return torch.softmax((q_head @ k_head.transpose(0, 1)) * SCALE, dim=-1) @ v_head


def run(rank: int, world_size: int, device: torch.device) -> None:
    assert HEADS == world_size, "this teaching layout assumes HEADS == world_size"
    assert SEQ % world_size == 0

    q_full, k_full, v_full = build_full_qkv()
    sl = slice(rank * SHARD, (rank + 1) * SHARD)
    q_local = q_full[sl].to(device)  # [S/n, HID]
    k_local = k_full[sl].to(device)
    v_local = v_full[sl].to(device)

    rank_print(rank, f"seq-sharded Q/K/V = {tuple(q_local.shape)} (full seq = {SEQ}, heads = {HEADS})")

    # [All-to-All] seq-sharded -> head-sharded (full sequence, our head group)
    q_head = all2all_seq_to_head(q_local, world_size, device)  # [S, HID/n]
    k_head = all2all_seq_to_head(k_local, world_size, device)
    v_head = all2all_seq_to_head(v_local, world_size, device)

    # Plain local attention over the full sequence for our own head(s)
    o_head = local_attention(q_head, k_head, v_head)           # [S, HID/n]

    # [All-to-All] head-sharded -> seq-sharded
    o_local = all2all_head_to_seq(o_head, world_size, device)  # [S/n, HID]

    # Correctness self-check: gather all sequence shards, compare to single-machine MHA.
    gathered = torch.empty(SEQ, HID, device=COMM_DEVICE)
    dist.all_gather_into_tensor(gathered, o_local.to(COMM_DEVICE).contiguous())
    ref = multihead_reference(q_full, k_full, v_full)
    err = (gathered - ref).abs().max().item()
    rank0_print(rank, f"Ulysses-SP output shape = {tuple(gathered.shape)} | max error vs single-machine = {err:.2e}")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
