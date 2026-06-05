"""Module 4 (DeepSeek) - DeepEP-style overlapped All-to-All dispatch + combine.

Principle in one line (DeepEP):
    1_naive_moe.py waits out the dispatch All-to-All AND the combine All-to-All, both blocking.
    2_lightning_moe.py overlapped the DISPATCH only. DeepEP (DeepSeek's expert-parallel comm
    library) goes further: it pipelines the ENTIRE MoE layer so that, across token chunks,
    a chunk's dispatch and the previous chunk's combine are BOTH in flight asynchronously
    while an expert is computing. Communication is almost fully hidden behind compute.

    DeepEP also distinguishes the network tiers: intra-node links (NVLink) are fast, inter-node
    links (RDMA/IB) are slow. This demo models that with two LinkProfiles so you can see why
    overlapping the slow inter-node combine matters most.

    Pipeline (prologue / steady / epilogue), per token chunk t:
        - fire async DISPATCH of chunk t+1   (inter-node, slow)
        - wait DISPATCH of chunk t, run expert(chunk t)   (compute hides t+1's dispatch)
        - fire async COMBINE of chunk t      (send results back, also slow)
        - one step later, wait COMBINE of chunk t-1
    so both All-to-Alls overlap with expert compute.

Your battle zone:
    - `deepep_moe_forward`: build the two-stage (dispatch + combine) overlapped pipeline.

Run: python 4_expert_parallel/4_deepseek_deepep.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from topology_sim import (  # noqa: F401
    LinkProfile,
    slow_all_to_all_single,
    slow_all_to_all_single_async,
)

from env_setup import (
    COMM_DEVICE,  # noqa: F401  (move comm tensors onto it)
    Timeline,
    bar,
    launch_teaching_cluster,
    rank0_print,
    rank_print,
)

WORLD_SIZE = 4
TOKENS_PER_EXPERT = 32
DIM = 64
NUM_CHUNKS = 4

# Two-tier network: intra-node NVLink (fast) vs inter-node RDMA (slow). Overlapping the
# slow inter-node transfers is where DeepEP earns its keep.
INTRA_NODE = LinkProfile(latency_s=0.002)   # NVLink-ish, 2ms
INTER_NODE = LinkProfile(latency_s=0.02)    # RDMA/IB-ish, 20ms


class Expert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def make_local_tokens(rank: int, world_size: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator().manual_seed(7000 + rank)
    return torch.randn(world_size * TOKENS_PER_EXPERT, DIM, generator=g).to(device)


def split_chunks(tokens: torch.Tensor, num_chunks: int) -> list[torch.Tensor]:
    assert tokens.shape[0] % num_chunks == 0, "token count must be divisible by NUM_CHUNKS"
    return list(tokens.chunk(num_chunks, dim=0))


def reference_forward(expert: Expert, tokens: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Synchronous dispatch -> expert -> combine, used only to check your overlapped version."""
    disp = torch.empty_like(tokens, device=COMM_DEVICE)
    slow_all_to_all_single(disp, tokens, INTER_NODE)
    out = expert(disp.to(device))
    comb = torch.empty_like(out, device=COMM_DEVICE)
    slow_all_to_all_single(comb, out, INTER_NODE)
    return comb.to(device)


def deepep_moe_forward(
    rank: int,
    world_size: int,
    expert: Expert,
    tokens: torch.Tensor,
    device: torch.device,
    timeline: Timeline,
) -> torch.Tensor:
    """Overlapped dispatch + combine pipeline over token chunks.

    ============================ YOUR BATTLE ZONE ============================
    chunks = split_chunks(tokens, NUM_CHUNKS)
    Keep small lists/dicts for in-flight work handles and buffers.

    Prologue: fire async dispatch for chunk 0:
        disp0 = empty_like(chunks[0]) on COMM_DEVICE
        d_work0 = slow_all_to_all_single_async(disp0, chunks[0], INTER_NODE)

    Steady (for t in range(NUM_CHUNKS)):
        - if t+1 exists: fire async dispatch for chunk t+1 (so it flies during compute)
        - d_work[t].wait()                                  # timeline span kind="comm"
        - out_t = expert(disp[t].to(device))                # timeline span kind="compute"
        - fire async COMBINE for out_t:
              comb_t = empty_like(out_t) on COMM_DEVICE
              c_work[t] = slow_all_to_all_single_async(comb_t, out_t, INTER_NODE)
        - if t-1 >= 0: c_work[t-1].wait(); stash comb[t-1] as a finished result
    Epilogue: wait the last combine, collect all comb[t], torch.cat along dim 0, return.

    The point: at any moment a dispatch (t+1) and a combine (t-1) are BOTH in flight while
    expert(t) computes -- both All-to-Alls hidden behind compute.
    ==========================================================================
    """
    # TODO(you): two-stage (dispatch + combine) overlapped pipeline with async All-to-All
    raise NotImplementedError("TODO: deepep_moe_forward -- overlap BOTH dispatch and combine across chunks")


def run(rank: int, world_size: int, device: torch.device) -> None:
    torch.manual_seed(300 + rank)
    expert = Expert().to(device)
    tokens = make_local_tokens(rank, world_size, device)

    rank_print(rank, f"local tokens = {tuple(tokens.shape)} | inter-node {INTER_NODE.latency_s * 1000:.0f}ms vs intra-node {INTRA_NODE.latency_s * 1000:.0f}ms | chunks = {NUM_CHUNKS}")

    timeline = Timeline()
    out = deepep_moe_forward(rank, world_size, expert, tokens, device, timeline)

    # Correctness self-check: overlapped result must equal the synchronous reference.
    ref = reference_forward(expert, tokens, device)
    err = (out.to(COMM_DEVICE) - ref.to(COMM_DEVICE)).abs().max().item()
    deepep_ms = timeline.makespan() * 1000
    rank0_print(rank, f"DeepEP makespan = {deepep_ms:.1f} ms | max error vs synchronous = {err:.2e}")

    # Rough comparison bar: a fully-serial layer pays ~2 * NUM_CHUNKS inter-node hops.
    serial_ms = 2 * NUM_CHUNKS * INTER_NODE.latency_s * 1000
    if rank == 0:
        worst = max(serial_ms, deepep_ms, 1.0)
        print()
        print("  -- inter-node All-to-All hidden behind compute --")
        print(f"  serial(est) |{bar(serial_ms, worst)}| {serial_ms:6.1f} ms")
        print(f"  deepep      |{bar(deepep_ms, worst)}| {deepep_ms:6.1f} ms")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
