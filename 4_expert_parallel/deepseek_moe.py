"""Module 4 (DeepSeek) - DeepSeekMoE: fine-grained + shared experts, auxiliary-loss-free balancing.

Two DeepSeek ideas in one line each:
    1. DeepSeekMoE architecture (V2/V3): use MANY small "fine-grained" routed experts plus a
       few always-on "shared" experts. Each token goes through the shared experts (capturing
       common knowledge) AND its top-K routed experts (capturing specialization):
           y = sum_s shared_s(x)  +  sum_{e in topK} gate_e * expert_e(x)
    2. Auxiliary-loss-free load balancing (V3): classic MoE adds an auxiliary loss to keep
       experts evenly loaded, which fights the main loss. DeepSeek-V3 instead keeps a per-expert
       BIAS b_e that is added to the routing score ONLY for top-K SELECTION (never for the gate
       value). After each step the bias is nudged from the GLOBAL load:
           b_e += gamma * sign(target_load - load_e)
       Overloaded experts get a lower bias (less likely to be picked), underloaded ones get a
       higher bias. No extra loss term, no gradient interference.

The distributed twist: the load `load_e` is GLOBAL across all ranks, so computing it needs an
All-Reduce(SUM) of every rank's local per-expert token counts. That All-Reduce is your second
battle zone -- it is what makes the balancer consistent across the whole (data/expert) parallel
group. (Experts/router are replicated here for clarity; the physical token All-to-All dispatch
is covered in naive_moe.py / lightning_moe.py / deepseek_deepep.py.)

Your battle zone (two TODOs):
    - `deepseek_route`: scores -> add bias -> top-K select -> normalized gates from RAW scores.
    - `update_expert_bias`: All-Reduce local counts into the global load, then nudge the bias.

Run: python 4_expert_parallel/deepseek_moe.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: F401  (you will write dist.all_reduce in a TODO)
import torch.nn as nn

from env_setup import COMM_DEVICE, launch_teaching_cluster, rank0_print, rank_print

WORLD_SIZE = 4
TOKENS = 32
DIM = 32
N_ROUTED = 8        # fine-grained routed experts (small + many)
N_SHARED = 1        # always-on shared experts
TOP_K = 2           # routed experts selected per token
STEPS = 6
BIAS_SPEED = 0.05   # gamma: how fast the aux-loss-free bias adapts


class Expert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def deepseek_route(
    scores: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-K routing with auxiliary-loss-free bias.

    Args:
        scores: routed-expert affinities, shape [T, N_ROUTED] (already softmax-normalized).
        bias:   per-expert selection bias, shape [N_ROUTED].
        top_k:  number of routed experts per token.
    Returns:
        idx:  selected expert indices, shape [T, top_k]
        gate: gating weights for the selected experts, shape [T, top_k], normalized to sum 1.

    ============================ YOUR BATTLE ZONE 1 ==========================
    Steps:
        1. Selection score = scores + bias (broadcast bias over the token dim).
           idx = torch.topk(scores + bias, top_k, dim=-1).indices      # [T, top_k]
        2. The GATE must come from the RAW scores (NOT the biased ones):
           gate = torch.gather(scores, 1, idx)                         # [T, top_k]
        3. Normalize the gates so they sum to 1 per token:
           gate = gate / gate.sum(dim=-1, keepdim=True)
        4. return idx, gate
    The bias steers WHICH experts are picked; it must never distort the gate magnitude.
    ==========================================================================
    """
    # TODO(you): biased top-K selection + raw-score normalized gates
    raise NotImplementedError("TODO: deepseek_route -- biased top-K selection with raw-score gates")


def update_expert_bias(
    bias: torch.Tensor,
    local_counts: torch.Tensor,
    world_size: int,
    target_load: float,
) -> torch.Tensor:
    """Auxiliary-loss-free bias update from the GLOBAL load (All-Reduce of local counts).

    ============================ YOUR BATTLE ZONE 2 ==========================
    Steps:
        1. Move local_counts to COMM_DEVICE, dist.all_reduce(SUM) to get the global per-expert
           load across all ranks, then move back.
        2. Nudge the bias: bias = bias + BIAS_SPEED * sign(target_load - global_counts)
           (underloaded -> bias up -> more likely selected next step; overloaded -> bias down).
        3. return the updated bias.
    ==========================================================================
    """
    # TODO(you): all_reduce local_counts -> global load, then nudge the bias by its sign
    raise NotImplementedError("TODO: update_expert_bias -- All-Reduce global load + sign-based bias nudge")


def deepseek_moe_forward(
    x: torch.Tensor,
    router_w: torch.Tensor,
    bias: torch.Tensor,
    shared: nn.ModuleList,
    routed: nn.ModuleList,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full DeepSeekMoE forward. Returns (output, local per-expert token counts)."""
    scores = torch.softmax(x @ router_w, dim=-1)              # [T, N_ROUTED]
    idx, gate = deepseek_route(scores, bias, TOP_K)           # your routing

    y = torch.zeros_like(x)
    for s in shared:                                         # always-on shared experts
        y = y + s(x)
    counts = torch.zeros(N_ROUTED, device=COMM_DEVICE)
    for slot in range(TOP_K):
        e_idx = idx[:, slot]                                 # [T]
        g = gate[:, slot : slot + 1]                         # [T, 1]
        for e in range(N_ROUTED):
            mask = e_idx == e
            if mask.any():
                y[mask] = y[mask] + g[mask] * routed[e](x[mask])
                counts[e] += int(mask.sum())
    return y, counts


def run(rank: int, world_size: int, device: torch.device) -> None:
    torch.manual_seed(0)  # identical replicated weights on every rank
    router_w = torch.randn(DIM, N_ROUTED, generator=torch.Generator().manual_seed(7)).to(device)
    shared = nn.ModuleList([Expert() for _ in range(N_SHARED)]).to(device)
    routed = nn.ModuleList([Expert() for _ in range(N_ROUTED)]).to(device)
    bias = torch.zeros(N_ROUTED, device=device)

    # Each rank has its own local tokens (its slice of the data-parallel batch).
    gx = torch.Generator().manual_seed(1000 + rank)
    x = torch.randn(TOKENS, DIM, generator=gx).to(device)
    target_load = TOKENS * world_size * TOP_K / N_ROUTED

    rank_print(rank, f"{N_ROUTED} routed + {N_SHARED} shared experts, top-{TOP_K} | target load/expert = {target_load:.0f}")

    for step in range(STEPS):
        _, counts = deepseek_moe_forward(x, router_w, bias, shared, routed)
        bias = update_expert_bias(bias, counts.clone(), world_size, target_load)
        if rank == 0:
            spread = counts.max().item() - counts.min().item()
            rank0_print(rank, f"step {step} | rank0 local counts = {counts.int().tolist()} | local spread = {int(spread)}")

    rank0_print(rank, "DeepSeekMoE done: bias-based balancing evens out expert load with no auxiliary loss.")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
