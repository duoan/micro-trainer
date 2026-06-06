"""Module 4 (Expert Parallelism) - Artificial network topology bottleneck simulator (Traffic Shaper).

Why do we need it?
    On a Mac, inter-process communication in unified memory is absurdly fast -- so
    fast you can't even see "communication cost" as a thing. But in real large-model
    training, MoE's All-to-All has to cross machines over an IB NIC, and the latency
    is very real and a genuine bottleneck.

    So we MANUFACTURE a bottleneck: inject a time.sleep into every All-to-All to
    simulate "cross-machine network latency". With this controllable latency, you can
    finally see the night-and-day difference between naive_moe (dumb waiting) and
    lightning_moe (Tile overlap).

This file is the "foundation tool" for you, ready to use as-is:
    - LinkProfile: describes the simulated link's latency parameters.
    - slow_all_to_all_single(...): a delay-injected All-to-All (synchronous, for naive).
    - slow_all_to_all_single_async(...): an async version returning (work, output) (for lightning's overlap).

All-to-All runs directly on the tensors' compute device (CPU under gloo, GPU under NCCL).
"""

from __future__ import annotations

import pathlib
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist  # noqa: E402


@dataclass(frozen=True)
class LinkProfile:
    """A performance profile for a simulated cross-machine link.

    Transfer time ~= latency + bytes / bandwidth. For clarity this project mainly
    uses a fixed latency.
    """

    latency_s: float = 0.02  # fixed latency per cross-machine transfer (default 20ms, deliberately exaggerated)
    bandwidth_bps: float = 0.0  # set >0 to add a per-byte transfer term; 0 ignores the bandwidth term

    def transfer_time(self, num_bytes: int) -> float:
        bw_term = (num_bytes / self.bandwidth_bps) if self.bandwidth_bps > 0 else 0.0
        return self.latency_s + bw_term


# Default link: 20ms latency. To worsen/ease the bottleneck, change this or pass a custom LinkProfile in a demo.
DEFAULT_LINK = LinkProfile(latency_s=0.02)


def _inject_delay(tensor: torch.Tensor, link: LinkProfile) -> None:
    """This is the source of the "artificial cross-machine latency" -- literally sleep for a while."""
    time.sleep(link.transfer_time(tensor.element_size() * tensor.nelement()))


def slow_all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    link: LinkProfile = DEFAULT_LINK,
) -> None:
    """Synchronous All-to-All with simulated cross-machine latency.

    Semantically equivalent to dist.all_to_all_single(output, input), it just sleeps
    first to simulate network latency. naive_moe uses this directly -- every
    communication dutifully waits out the full latency while the compute units idle.
    """
    _inject_delay(input, link)
    dist.all_to_all_single(output, input)


def slow_all_to_all_single_async(
    output: torch.Tensor,
    input: torch.Tensor,
    link: LinkProfile = DEFAULT_LINK,
):
    """Async All-to-All: returns a work handle immediately, without blocking.

    This is the key to lightning_moe's "comm-compute overlap":
        work = slow_all_to_all_single_async(out_tile, in_tile)
        ...        # meanwhile do the expert compute of the PREVIOUS tile, hiding the latency
        work.wait()  # only wait when out_tile is actually needed

    Note: truly modeling both "latency" AND "async non-blocking" isn't trivial (a real
    async NIC uses hardware DMA). The compromise this teaching implementation takes:
    fire a real non-blocking all_to_all with async_op=True, and book the "latency" onto
    the returned handle, to be settled by the caller at wait() time. So the SHAPE of the
    overlap is real, while the AMOUNT of latency is simulated.
    """
    work = dist.all_to_all_single(output, input, async_op=True)
    nbytes = input.element_size() * input.nelement()
    return _DelayedWork(work, link.transfer_time(nbytes))


class _DelayedWork:
    """Wraps a work handle; wait() tops up the simulated latency so the overlap is observable."""

    def __init__(self, work, delay_s: float) -> None:
        self._work = work
        self._delay_s = delay_s
        self._t0 = time.perf_counter()

    def wait(self) -> None:
        # If the caller did other compute before this, that time "ate" part of the
        # latency -- which is exactly the overlap.
        elapsed = time.perf_counter() - self._t0
        remaining = self._delay_s - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._work.wait()


if __name__ == "__main__":
    # Quick self-check: run this file as a script to verify delay injection roughly
    # matches expectation (single process, no rendezvous needed).
    link = LinkProfile(latency_s=0.02)
    t = torch.randn(1024)
    start = time.perf_counter()
    _inject_delay(t, link)
    print(f"injected delay ~= {(time.perf_counter() - start) * 1000:.1f} ms (expected ~20ms)")
