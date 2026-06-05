"""Module 8 - Capstone II: N-D ("6D") Mesh Parallelism -- compose every axis on one mesh.

Principle in one line (after "Visualizing 6D Mesh Parallelism"):
    Each parallelism you built in modules 1-7 is ONE axis of a single device mesh. Real
    frontier training stacks all of them: the canonical hierarchy, innermost (closest,
    fastest links) -> outermost (furthest), is

        [ TP , EP , CP , FS , DP , PP ]

    TP sits closest (low-latency intra-node All-Reduce), PP furthest (rare P2P across
    islands). A rank's flat id decomposes into a coordinate along every axis, and each
    axis owns a set of process groups -- the "lines" of the mesh. During a training step
    each axis's group "flashes" when its collective fires:

        TP -> All-Reduce per layer          (2_tensor_parallel/)
        EP -> All-to-All dispatch/combine   (4_expert_parallel/)
        CP -> All-Gather K/V (or ring)      (6_context_parallel/1_ring_attention.py)
        FS -> All-Gather params + Reduce-Scatter grads  (1_data_parallel/3_fsdp_demo.py)
        DP -> All-Reduce grads              (1_data_parallel/1_ddp_demo.py)
        PP -> P2P send/recv                 (3_pipeline_parallel/)

    This demo doesn't re-derive those collectives (the other modules already do). It builds
    the FULL mesh, prints every rank's 6D coordinate, and then walks the axes in hierarchy
    order, "pinging" each axis's group (a coordinate All-Reduce that proves the group is
    wired correctly) while flashing the mesh -- the terminal analogue of the 6D blog visual.

Your battle zone:
    - `ping_axis`: All-Reduce a coordinate probe over an axis's process group, so we can
      verify the mesh (and you feel exactly which ranks a given axis talks to).

Tip: bump the MESH sizes (and WORLD_SIZE follows) to light up more axes -- but every extra
factor multiplies the process count, so keep it small on a laptop.

Run: python 8_hybrid_parallel/2_nd_mesh_parallel.py
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: F401  (you will write dist.all_reduce in the TODO)

from env_setup import (
    COMM_DEVICE,  # noqa: F401  (you will use COMM_DEVICE in the ping_axis TODO)
    C,
    banner,
    build_mesh,
    launch_teaching_cluster,
    mesh_summary,
    rank0_print,
    rank_print,
    render_mesh,
)

# Canonical hierarchy, innermost (closest) -> outermost (furthest).
CANONICAL = ["TP", "EP", "CP", "FS", "DP", "PP"]

# The real collective + the module that hand-writes it, for each axis (for the trace log).
AXIS_INFO = {
    "TP": "All-Reduce per layer        (2_tensor_parallel/)",
    "EP": "All-to-All dispatch/combine (4_expert_parallel/)",
    "CP": "All-Gather K/V or ring      (6_context_parallel/1_ring_attention.py)",
    "FS": "All-Gather + Reduce-Scatter (1_data_parallel/3_fsdp_demo.py)",
    "DP": "All-Reduce grads            (1_data_parallel/1_ddp_demo.py)",
    "PP": "P2P send/recv               (3_pipeline_parallel/)",
}

# Edit me! Product must stay == the spawned world. Default lights up TP, CP, DP (8 ranks).
# To exercise more axes, e.g. {"TP":2,"EP":2,"CP":1,"FS":2,"DP":2,"PP":1} -> 16 ranks.
MESH = {"TP": 2, "EP": 1, "CP": 2, "FS": 1, "DP": 2, "PP": 1}
WORLD_SIZE = math.prod(MESH.values())


def ping_axis(mesh, axis: str) -> float:
    """All-Reduce a coordinate probe over `axis`'s group to prove the group is wired right.

    Each group along an axis holds exactly one rank per coordinate value (0..size-1), so
    summing every member's coordinate must give 0+1+...+(size-1) = size*(size-1)/2.

    ============================ YOUR BATTLE ZONE ============================
    group = mesh.groups[axis]
    probe = torch.full((1,), float(mesh.coords[axis]), device=COMM_DEVICE)
    dist.all_reduce(probe, op=dist.ReduceOp.SUM, group=group)
    return probe.item()
    ==========================================================================
    """
    # TODO(you): all_reduce the coordinate probe over mesh.groups[axis]
    raise NotImplementedError("TODO: ping_axis -- All-Reduce a coordinate probe over the axis's process group")


def run(rank: int, world_size: int, device: torch.device) -> None:
    mesh = build_mesh(MESH)

    rank0_print(rank, banner("N-D (6D) Mesh Parallelism", color=C.BOLD + C.CYAN))
    rank0_print(rank, mesh_summary(mesh))
    if rank == 0:
        print(f"{C.BOLD}axis -> real collective (and the module that hand-writes it):{C.RESET}")
        for a in CANONICAL:
            state = f"size={MESH[a]}" if MESH[a] > 1 else "size=1 (degenerate)"
            print(f"  {a:<3} {state:<18} {AXIS_INFO[a]}")

    rank_print(rank, "coord " + " ".join(f"{a}={mesh.coords[a]}" for a in CANONICAL))

    # Walk the hierarchy: ping + flash each active axis's groups.
    for axis in CANONICAL:
        if MESH[axis] == 1:
            rank0_print(rank, f"{C.GREY}{axis}: degenerate (size 1) -> no communication{C.RESET}")
            continue
        got = ping_axis(mesh, axis)
        expected = MESH[axis] * (MESH[axis] - 1) / 2.0
        ok = abs(got - expected) < 1e-6
        rank0_print(rank, render_mesh(mesh, highlight_axis=axis))
        rank_print(rank, f"{axis} ping={got} (expected {expected}) {'OK' if ok else 'MISMATCH'}")

    rank0_print(rank, "6D mesh done: TP closest ... PP furthest; each axis pinged its own process group.")


if __name__ == "__main__":
    launch_teaching_cluster(world_size=WORLD_SIZE, func=run)
