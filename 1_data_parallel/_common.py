"""Shared scaffolding for the Module 1 data-parallel demos.

The three demos (`1_ddp.py`, `2_fsdp.py`, `3_hsdp.py`) each wrap ONE hand-written
strategy with several selectable *levels*. Everything that is NOT the distributed core
-- the toy model, the dataset, flatten/​unflatten helpers, and the little CLI launcher --
lives here so the demos don't repeat it.

Each demo exposes:
    LEVELS = [...]                      # the levels you can implement / run
    def run(rank, world_size, device, level): ...
    if __name__ == "__main__": cli(LEVELS, run)

and you run a single level with e.g. `python 1_data_parallel/1_ddp.py overlap`.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from env_setup import launch_teaching_cluster

WORLD_SIZE = 4
STEPS = 5
GLOBAL_BATCH = 64
IN_DIM, HIDDEN, OUT_DIM = 16, 32, 1


class TinyMLP(nn.Module):
    """A 2-layer MLP -- small enough to read every gradient, real enough to train."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, OUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_global_dataset(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The full global dataset, identical on every rank via a fixed seed."""
    g = torch.Generator().manual_seed(42)
    x = torch.randn(GLOBAL_BATCH, IN_DIM, generator=g)
    true_w = torch.randn(IN_DIM, OUT_DIM, generator=g)
    y = x @ true_w + 0.1 * torch.randn(GLOBAL_BATCH, OUT_DIM, generator=g)
    return x.to(device), y.to(device)


def shard_for(
    x: torch.Tensor, y: torch.Tensor, rank: int, world_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's contiguous slice of the global batch."""
    per = GLOBAL_BATCH // world_size
    sl = slice(rank * per, (rank + 1) * per)
    return x[sl], y[sl]


def make_shard(
    rank: int, world_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience: build the global dataset and return just this rank's shard."""
    x, y = build_global_dataset(device)
    return shard_for(x, y, rank, world_size)


# --------------------------------------------------------------------------- #
# Flatten / unflatten a model <-> one flat vector. FSDP and HSDP both shard the
# flattened parameter/gradient vector, so these live here once.
# --------------------------------------------------------------------------- #
def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def flatten_grads(model: nn.Module) -> torch.Tensor:
    return torch.cat(
        [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
    )


def load_flat_into_model(model: nn.Module, flat: torch.Tensor) -> None:
    off = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[off : off + n].view_as(p))
        off += n


def pad_to_multiple(flat: torch.Tensor, n: int) -> torch.Tensor:
    pad = (-flat.numel()) % n
    return torch.cat([flat, flat.new_zeros(pad)]) if pad else flat


def cli(levels: list[str], run: Callable[..., None], world_size: int = WORLD_SIZE) -> None:
    """Parse ``python <demo>.py [level]`` and launch the cluster for the chosen level.

    With no argument it runs the first (simplest) level. The level string is forwarded
    to ``run(rank, world_size, device, level)`` inside every spawned rank.
    """
    level = sys.argv[1] if len(sys.argv) > 1 else levels[0]
    if level not in levels:
        raise SystemExit(f"unknown level {level!r}; choose from {levels}")
    launch_teaching_cluster(world_size, run, level)
