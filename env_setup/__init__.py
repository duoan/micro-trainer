"""env_setup -- the foundation layer of micro-trainer.

This package wraps up all the dirty, tedious plumbing of "simulating a GPU
cluster with multiple processes on a single box", so that in every module you
can focus purely on hand-writing the distributed operators and scheduling logic.

It auto-adapts to whatever hardware you run on:

    * NVIDIA GPU box  -> backend "nccl", each rank binds its own `cuda:i`.
    * Mac / CPU box    -> backend "gloo", everything runs on `cpu`.

(We deliberately do NOT use Apple's MPS: the toy models are tiny so MPS buys
nothing, and gloo cannot run collectives on MPS tensors -- which would force a
constant MPS<->CPU shuffle that only obscures the real lessons. Plain CPU keeps
``device == COMM_DEVICE``, so the comm pattern is identical to the GPU/NCCL case.)

You can force a choice with the env var ``MICRO_TRAINER_BACKEND=gloo|nccl``.

It exposes a few core things:

    launch_teaching_cluster(world_size, func)
        Spin up `world_size` processes, each pretending to be one "GPU rank".
        The backend is auto-detected (override with `backend=...`).

    bind_device(rank)
        Bind a compute device to each rank: `cuda:rank` on a GPU box, else `cpu`.

    COMM_DEVICE
        A very important constant: where collective tensors must live.
        * Under NCCL it equals the rank's CUDA device, so the golden-rule move
          ``tensor.to(COMM_DEVICE)`` is a free no-op (NCCL talks GPU directly).
        * Under gloo it is `cpu` -- and since compute also runs on `cpu`, the move
          is again a no-op. So "compute on device, ``.to(COMM_DEVICE)`` before
          communicating" stays in the code purely for portability: it's the bridge
          that does real work only if you force gloo on top of CUDA tensors.

It also ships a set of hand-rolled terminal color / printing helpers so you can
draw flashy Timeline dashboards.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import math
import os
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

__all__ = [
    "launch_teaching_cluster",
    "bind_device",
    "default_backend",
    "COMM_DEVICE",
    "C",
    "rank_print",
    "rank0_print",
    "banner",
    "bar",
    "Timeline",
    "render_gantt",
    "DeviceMesh",
    "build_mesh",
    "mesh_summary",
    "render_mesh",
]


# --------------------------------------------------------------------------- #
# Backend + communication-device auto-detection (NCCL on GPU, gloo on Mac/CPU)
# --------------------------------------------------------------------------- #
def default_backend() -> str:
    """Pick the communication backend for this machine.

    Order of preference: an explicit ``MICRO_TRAINER_BACKEND`` env override,
    then NCCL when CUDA + NCCL are available (real GPU box), otherwise gloo
    (Apple Silicon / CPU).
    """
    forced = os.environ.get("MICRO_TRAINER_BACKEND", "").strip().lower()
    if forced in ("nccl", "gloo"):
        return forced
    if torch.cuda.is_available() and dist.is_nccl_available():
        return "nccl"
    return "gloo"


# Where collective tensors must live. Under NCCL that is the current CUDA device
# (so `tensor.to(COMM_DEVICE)` is a free no-op and collectives run on the GPU);
# under gloo it is cpu -- and since we compute on cpu too, that move is also a
# no-op. `torch.device("cuda")` (no index) resolves to whatever
# `torch.cuda.set_device` picked for this rank, so one constant works everywhere.
COMM_DEVICE = torch.device("cuda") if default_backend() == "nccl" else torch.device("cpu")


# --------------------------------------------------------------------------- #
# Terminal colors: hand-rolled ANSI, no third-party deps, handy for dashboards
# --------------------------------------------------------------------------- #
class C:
    """A tiny set of ANSI color constants. Usage: print(f"{C.RED}hi{C.RESET}")."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GREY = "\033[90m"

    # Assign each rank a fixed color so interleaved logs are easy to tell apart.
    _RANK_COLORS = [GREEN, CYAN, YELLOW, MAGENTA, BLUE, RED]

    @classmethod
    def rank(cls, rank: int) -> str:
        return cls._RANK_COLORS[rank % len(cls._RANK_COLORS)]


def _ts() -> str:
    return _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def rank_print(rank: int, msg: str) -> None:
    """Print with a per-rank color and timestamp, so interleaved multi-process logs stay readable."""
    color = C.rank(rank)
    print(f"{color}[{_ts()}][rank{rank}]{C.RESET} {msg}", flush=True)


def rank0_print(rank: int, msg: str) -> None:
    """Only rank 0 prints -- use for global summaries / dashboards to avoid log spam."""
    if rank == 0:
        print(msg, flush=True)


def banner(text: str, color: str = C.BOLD) -> str:
    """Build an eye-catching divider banner, good for module headers / dashboard titles."""
    line = "=" * max(len(text) + 4, 40)
    return f"{color}{line}\n  {text}\n{line}{C.RESET}"


def bar(value: float, total: float, width: int = 40, fill: str = "#", empty: str = "-") -> str:
    """Render a ratio as an ASCII progress / histogram bar, for performance comparison dashboards."""
    total = total or 1.0
    n = int(round(width * min(max(value / total, 0.0), 1.0)))
    return fill * n + empty * (width - n)


# --------------------------------------------------------------------------- #
# Timeline dashboard scaffolding
# --------------------------------------------------------------------------- #
# This is the "foundation" for the timeline dashboards in modules 3 and 4. It
# only records "an activity happened on which rank, with what label, from when
# to when". The truly flashy rendering (1F1B conveyor belt, comm-bubble
# comparison bars) is yours to extend -- here is a minimal working version.
@dataclass
class _Event:
    rank: int
    label: str
    start: float
    end: float
    kind: str = "compute"  # "compute" | "comm" | "bubble"


@dataclass
class Timeline:
    """Minimal event recorder. Wrap an operation with `with timeline.span(...)` to auto-time it."""

    t0: float = field(default_factory=time.perf_counter)
    events: list[_Event] = field(default_factory=list)

    @contextlib.contextmanager
    def span(self, rank: int, label: str, kind: str = "compute"):
        start = time.perf_counter() - self.t0
        try:
            yield
        finally:
            end = time.perf_counter() - self.t0
            self.events.append(_Event(rank, label, start, end, kind))

    def add(self, rank: int, label: str, start: float, end: float, kind: str = "compute") -> None:
        self.events.append(_Event(rank, label, start, end, kind))

    def makespan(self) -> float:
        return max((e.end for e in self.events), default=0.0)


_KIND_GLYPH = {"compute": "#", "comm": "=", "bubble": "."}
_KIND_COLOR = {"compute": C.GREEN, "comm": C.YELLOW, "bubble": C.GREY}


def render_gantt(timeline: Timeline, world_size: int, width: int = 60) -> str:
    """Render a Timeline as an ASCII Gantt chart (one row per rank).

    This is the most bare-bones version: it places each event on its rank's row
    proportionally to time. Want a labeled [F1][F2][B1] conveyor matrix? Rewrite it!
    """
    span = timeline.makespan() or 1.0
    lines = [f"{C.BOLD}Timeline (makespan = {span * 1000:.1f} ms){C.RESET}"]
    for r in range(world_size):
        row = [" "] * width
        for e in timeline.events:
            if e.rank != r:
                continue
            lo = int(e.start / span * width)
            hi = max(int(e.end / span * width), lo + 1)
            glyph = _KIND_GLYPH.get(e.kind, "#")
            color = _KIND_COLOR.get(e.kind, C.GREEN)
            for i in range(lo, min(hi, width)):
                row[i] = f"{color}{glyph}{C.RESET}"
        lines.append(f"{C.rank(r)}rank{r}{C.RESET} |" + "".join(row) + "|")
    legend = f"{C.GREEN}# compute{C.RESET}  {C.YELLOW}= comm{C.RESET}  {C.GREY}. bubble{C.RESET}"
    lines.append(legend)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# N-D device mesh: compose TP / EP / CP / FS / DP / PP onto one mesh
# --------------------------------------------------------------------------- #
# Inspired by "Visualizing 6D Mesh Parallelism". You hand `build_mesh` an ORDERED
# dict of axis sizes, innermost (closest, fastest links) -> outermost (furthest):
# the canonical order is TP, EP, CP, FS, DP, PP. It returns, for THIS rank, its
# coordinate along every axis plus the process group it belongs to per axis.
#
# Rank decomposition (innermost axis varies fastest, so its ranks are contiguous):
#     rank = sum_a coord[a] * stride[a],  stride grows from the innermost axis out.
# A process group for axis `a` = the ranks that share every OTHER coordinate and
# differ only along `a` (the "line" along that axis).
@dataclass
class DeviceMesh:
    """A composed N-D device mesh. `groups[axis]` is None for degenerate (size-1) axes."""

    sizes: dict[str, int]
    coords: dict[str, int]
    groups: dict[str, Any]
    strides: dict[str, int]
    rank: int
    world_size: int

    def coord_of(self, rank: int) -> dict[str, int]:
        return {a: (rank // self.strides[a]) % self.sizes[a] for a in self.sizes}

    def group_base(self, rank: int, axis: str) -> int:
        """Lowest rank in `rank`'s line along `axis` (its group's representative)."""
        c = (rank // self.strides[axis]) % self.sizes[axis]
        return rank - c * self.strides[axis]

    def group_members(self, rank: int, axis: str) -> list[int]:
        base = self.group_base(rank, axis)
        return [base + i * self.strides[axis] for i in range(self.sizes[axis])]


def build_mesh(sizes: dict[str, int]) -> DeviceMesh:
    """Build an N-D device mesh and this rank's per-axis process groups.

    `sizes` is an ORDERED dict, innermost -> outermost, e.g.
        {"TP": 2, "EP": 1, "CP": 2, "FS": 1, "DP": 2, "PP": 1}
    Its product must equal the world size. Call this inside a `run(...)` after the
    process group is initialized; every rank must call it with the SAME `sizes`
    (dist.new_group is collective). Size-1 axes are skipped (group = None).
    """
    rank = dist.get_rank()
    world = dist.get_world_size()
    total = math.prod(sizes.values())
    if total != world:
        raise ValueError(f"mesh sizes {sizes} (product {total}) != world_size {world}")

    axes = list(sizes.keys())
    strides: dict[str, int] = {}
    s = 1
    for a in axes:
        strides[a] = s
        s *= sizes[a]

    def coord(r: int) -> dict[str, int]:
        return {a: (r // strides[a]) % sizes[a] for a in axes}

    my = coord(rank)
    groups: dict[str, Any] = {}
    for a in axes:
        if sizes[a] == 1:
            groups[a] = None
            continue
        # group ranks by "all coords except a"; create each line collectively
        lines: dict[tuple, list[int]] = {}
        for r in range(world):
            c = coord(r)
            key = tuple((b, c[b]) for b in axes if b != a)
            lines.setdefault(key, []).append(r)
        my_key = tuple((b, my[b]) for b in axes if b != a)
        my_group = None
        for key in sorted(lines.keys()):
            g = dist.new_group(ranks=lines[key])
            if key == my_key:
                my_group = g
        groups[a] = my_group

    return DeviceMesh(sizes, my, groups, strides, rank, world)


def mesh_summary(mesh: DeviceMesh) -> str:
    """A per-rank textual summary: each axis's size, this rank's coord, and its group."""
    lines = [f"{C.BOLD}Device mesh (world={mesh.world_size}), innermost -> outermost:{C.RESET}"]
    for a, n in mesh.sizes.items():
        if n == 1:
            lines.append(f"  {C.GREY}{a:<3} size=1   (degenerate, no communication){C.RESET}")
        else:
            members = mesh.group_members(mesh.rank, a)
            lines.append(f"  {a:<3} size={n}   rank{mesh.rank} coord={mesh.coords[a]}   group={members}")
    return "\n".join(lines)


def render_mesh(mesh: DeviceMesh, highlight_axis: str | None = None) -> str:
    """Render the whole world as a strip of rank cells; 'flash' one axis's groups in color.

    When `highlight_axis` is given, ranks are colored by which group-along-that-axis they
    belong to -- so you can see the mesh partition that a given collective communicates over.
    """
    palette = [C.GREEN, C.CYAN, C.YELLOW, C.MAGENTA, C.BLUE, C.RED]
    active = bool(highlight_axis) and mesh.sizes.get(highlight_axis, 1) > 1
    ordinal: dict[int, int] = {}
    if active:
        bases = sorted({mesh.group_base(r, highlight_axis) for r in range(mesh.world_size)})
        ordinal = {b: i for i, b in enumerate(bases)}
    cells = []
    for r in range(mesh.world_size):
        if active:
            color = palette[ordinal[mesh.group_base(r, highlight_axis)] % len(palette)]
            cells.append(f"{color}{r:2d}{C.RESET}")
        else:
            cells.append(f"{C.GREY}{r:2d}{C.RESET}")
    head = f"{C.BOLD}flash {highlight_axis}{C.RESET}: groups along {highlight_axis} communicate\n" if highlight_axis else ""
    return head + "[" + " ".join(cells) + "]"


# --------------------------------------------------------------------------- #
# Device binding
# --------------------------------------------------------------------------- #
def bind_device(rank: int) -> torch.device:
    """Bind a compute device to the current rank.

    * GPU box: each process owns one GPU. We call `torch.cuda.set_device(i)` and
      return `cuda:i` (i = rank, wrapped by device_count so it still runs when
      you simulate more ranks than you have GPUs -- handy for teaching).
    * Otherwise (Mac / CPU box): plain CPU. We intentionally skip MPS -- the toy
      models gain nothing from it, and gloo can't communicate MPS tensors anyway.
    """
    if torch.cuda.is_available():
        idx = rank % torch.cuda.device_count()
        torch.cuda.set_device(idx)
        return torch.device("cuda", idx)
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Cluster launcher
# --------------------------------------------------------------------------- #
def _find_free_port(start: int = 29500, tries: int = 200) -> int:
    """Find a free port starting from `start`, to avoid 'Address already in use' on re-runs."""
    for offset in range(tries):
        port = start + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}..{start + tries}")


def _worker(
    rank: int,
    world_size: int,
    func: Callable[..., Any],
    backend: str,
    master_addr: str,
    master_port: str,
    user_args: tuple[Any, ...],
) -> None:
    """Entry point of each child process: init process group -> bind device -> run user code -> clean up."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Bind the device first: NCCL wants `torch.cuda.set_device` set before init
    # so each rank initializes on its own GPU instead of all piling onto cuda:0.
    device = bind_device(rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    try:
        func(rank, world_size, device, *user_args)
    finally:
        # Barrier so everyone finishes, then destroy the group, so nobody exits
        # early and leaves peers hanging / erroring on a collective.
        with contextlib.suppress(Exception):
            dist.barrier()
        dist.destroy_process_group()


def launch_teaching_cluster(
    world_size: int,
    func: Callable[..., Any],
    *args: Any,
    backend: str | None = None,
    master_addr: str = "127.0.0.1",
    base_port: int = 29500,
) -> None:
    """Spin up `world_size` processes locally to simulate a `world_size`-GPU cluster.

    Args:
        world_size: Number of ranks to simulate (~ "how many GPUs").
        func: The function each rank runs, with the convention
              ``func(rank: int, world_size: int, device: torch.device, *args)``.
        *args: Extra positional args forwarded to `func`.
        backend: Communication backend. Defaults to auto-detection
              (NCCL on a GPU box, gloo on Mac/CPU); pass "gloo"/"nccl" to force.
        master_addr / base_port: Rendezvous address; a free port is picked
              automatically, so there is nothing to configure by hand.

    Usage (inside each demo's __main__)::

        from env_setup import launch_teaching_cluster

        def run(rank, world_size, device):
            ...  # your distributed core logic

        if __name__ == "__main__":
            launch_teaching_cluster(world_size=4, func=run)
    """
    backend = backend or default_backend()
    port = str(_find_free_port(base_port))

    if torch.cuda.is_available():
        accel = f"cuda x{torch.cuda.device_count()}"
    else:
        accel = "cpu"

    print(
        banner(
            f"launch_teaching_cluster | world_size={world_size} "
            f"backend={backend} compute={accel} | {master_addr}:{port}",
            color=C.BOLD + C.CYAN,
        )
    )
    if backend == "nccl" and torch.cuda.device_count() < world_size:
        print(
            f"{C.YELLOW}Only {torch.cuda.device_count()} GPU(s) for {world_size} ranks; "
            f"ranks will share GPUs round-robin (fine for small teaching runs).{C.RESET}"
        )

    # Use spawn to start child processes; join=True blocks until all ranks exit.
    mp.spawn(
        _worker,
        args=(world_size, func, backend, master_addr, port, args),
        nprocs=world_size,
        join=True,
    )

    print(f"{C.GREEN}All ranks exited; process group cleanly destroyed.{C.RESET}")
