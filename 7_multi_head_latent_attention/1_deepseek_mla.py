"""Module 7 (DeepSeek) - MLA: Multi-head Latent Attention (low-rank KV compression).

Principle in one line (DeepSeek-V2):
    Standard multi-head attention caches a full K and V per token (size H*Dh each), which
    dominates inference memory. MLA instead projects the hidden state DOWN to a small
    LATENT vector c_KV (dim d_c << H*Dh), and that latent is the ONLY thing cached. K and V
    are reconstructed on the fly by UP-projecting the latent:
        c_KV = x @ W_DKV            # [T, d_c]  -- the compressed KV cache
        K    = (c_KV @ W_UK)        # [T, H*Dh]
        V    = (c_KV @ W_UV)        # [T, H*Dh]
    This shrinks the KV cache by a large factor (d_c vs H*Dh) with almost no quality loss.

    In this teaching demo we also shard the heads across ranks (tensor parallel over heads,
    a la Module 2): each rank up-projects/attends only its OWN head group, then an All-Gather
    concatenates the per-head outputs into the full result. The shared latent c_KV is cheap,
    so every rank recomputes it locally.

Your battle zone:
    - `compress_and_project`: compute the latent c_KV, then up-project to this rank's local
      Q, K, V head group.

Run: python 7_multi_head_latent_attention/1_deepseek_mla.py
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
SEQ, D_MODEL = 12, 32
HEADS, HEAD_DIM = 4, 8       # HEADS == WORLD_SIZE so each rank handles 1 head
LATENT_DIM = 16              # d_c: the compressed KV cache dimension (<< HEADS*HEAD_DIM = 32)
HEADS_PER_RANK = HEADS // WORLD_SIZE
SCALE = 1.0 / math.sqrt(HEAD_DIM)


def build_full_weights() -> dict[str, torch.Tensor]:
    """All ranks build identical weights for cross-checking against single-machine MLA."""
    g = torch.Generator().manual_seed(5)
    return {
        "W_DKV": torch.randn(D_MODEL, LATENT_DIM, generator=g),       # down-project to latent
        "W_UK": torch.randn(LATENT_DIM, HEADS * HEAD_DIM, generator=g),  # up-project latent -> K
        "W_UV": torch.randn(LATENT_DIM, HEADS * HEAD_DIM, generator=g),  # up-project latent -> V
        "W_Q": torch.randn(D_MODEL, HEADS * HEAD_DIM, generator=g),   # query projection
    }


def head_slice(rank: int) -> slice:
    """Columns of the (H*Dh)-wide projections that belong to this rank's head group."""
    width = HEADS_PER_RANK * HEAD_DIM
    return slice(rank * width, (rank + 1) * width)


def compress_and_project(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MLA core: down-project x to the latent c_KV, then up-project to this rank's Q, K, V.

    ============================ YOUR BATTLE ZONE ============================
    Let sl = head_slice(rank) select this rank's head-group columns. Steps:
        1. c_kv = x @ w["W_DKV"]                       # [SEQ, LATENT_DIM]  (the cached latent)
        2. k = (c_kv @ w["W_UK"][:, sl]).view(SEQ, HEADS_PER_RANK, HEAD_DIM)
        3. v = (c_kv @ w["W_UV"][:, sl]).view(SEQ, HEADS_PER_RANK, HEAD_DIM)
        4. q = (x   @ w["W_Q"][:, sl]).view(SEQ, HEADS_PER_RANK, HEAD_DIM)
        5. return q, k, v
    The key idea: only c_kv (dim LATENT_DIM) would be cached, NOT the full k/v.
    No communication here -- it is local compute on `device`.
    ==========================================================================
    """
    # TODO(you): down-project to latent, then up-project to local Q/K/V head group
    raise NotImplementedError("TODO: compress_and_project -- hand-write MLA latent down/up projection")


def local_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Per-head attention for this rank's heads. q/k/v: [SEQ, HEADS_PER_RANK, HEAD_DIM] -> [SEQ, hpr*Dh]."""
    outs = []
    for h in range(HEADS_PER_RANK):
        s = (q[:, h] @ k[:, h].transpose(0, 1)) * SCALE
        outs.append(torch.softmax(s, dim=-1) @ v[:, h])
    return torch.cat(outs, dim=-1)


def gather_heads(o_local: torch.Tensor, world_size: int, device: torch.device) -> torch.Tensor:
    """All-Gather the per-rank head outputs and concatenate into the full [SEQ, HEADS*HEAD_DIM]."""
    width = HEADS_PER_RANK * HEAD_DIM
    buf = torch.empty(world_size * SEQ * width, device=COMM_DEVICE)
    dist.all_gather_into_tensor(buf, o_local.reshape(-1).to(COMM_DEVICE).contiguous())
    parts = buf.reshape(world_size, SEQ, width)
    return torch.cat([parts[r] for r in range(world_size)], dim=-1).to(device)


def run(rank: int, world_size: int, device: torch.device) -> None:
    assert HEADS == world_size, "this teaching layout assumes HEADS == world_size"

    w = {k: t.to(device) for k, t in build_full_weights().items()}
    gx = torch.Generator().manual_seed(99)
    x = torch.randn(SEQ, D_MODEL, generator=gx).to(device)

    rank_print(rank, f"latent dim d_c = {LATENT_DIM} vs full KV per token = {HEADS * HEAD_DIM} (cache shrinks ~{HEADS * HEAD_DIM / LATENT_DIM:.1f}x)")

    q, k, v = compress_and_project(x, w, rank, device)
    o_local = local_attention(q, k, v)               # [SEQ, hpr*Dh]
    out = gather_heads(o_local, world_size, device)  # [SEQ, HEADS*HEAD_DIM]

    # Correctness self-check: full single-machine MLA reference.
    cw = build_full_weights()
    c_kv = x.to(COMM_DEVICE) @ cw["W_DKV"]
    K = (c_kv @ cw["W_UK"]).view(SEQ, HEADS, HEAD_DIM)
    V = (c_kv @ cw["W_UV"]).view(SEQ, HEADS, HEAD_DIM)
    Q = (x.to(COMM_DEVICE) @ cw["W_Q"]).view(SEQ, HEADS, HEAD_DIM)
    ref = torch.empty(SEQ, HEADS * HEAD_DIM)
    for h in range(HEADS):
        s = (Q[:, h] @ K[:, h].transpose(0, 1)) * SCALE
        ref[:, h * HEAD_DIM : (h + 1) * HEAD_DIM] = torch.softmax(s, dim=-1) @ V[:, h]
    err = (out.to(COMM_DEVICE) - ref).abs().max().item()
    rank0_print(rank, f"MLA output shape = {tuple(out.shape)} | max error vs single-machine = {err:.2e}")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
