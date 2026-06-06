# Topology Simulator (Traffic Shaper)

> Manufactures cross-machine network latency on a Mac so MoE All-to-All communication cost is visible and comm-compute overlap can be measured.

## Why

On a Mac, inter-process communication over unified memory is absurdly fast — so fast that "communication cost" barely registers. Real large-model MoE training is different: **All-to-All** dispatch and combine cross machines over an InfiniBand NIC, and that latency is a genuine bottleneck.

The naive and lightning MoE demos need a controllable delay to expose the gap between **serial waiting** and **Tile overlap**. This module is the foundation tool: it injects modeled latency around real `dist.all_to_all_single` calls so you can see the night-and-day difference on a laptop.

Everything — compute and collectives — runs on the same **`device`** (gloo communicates CPU tensors, NCCL communicates GPU tensors directly).

## What it provides

### `LinkProfile`

Describes a simulated cross-machine link using the classic **alpha–beta** cost model:

```python
@dataclass(frozen=True)
class LinkProfile:
    latency_s: float = 0.02      # fixed latency per transfer (alpha)
    bandwidth_bps: float = 0.0   # bytes/sec; 0 ignores the bandwidth term (beta)

    def transfer_time(self, num_bytes: int) -> float:
        bw_term = (num_bytes / self.bandwidth_bps) if self.bandwidth_bps > 0 else 0.0
        return self.latency_s + bw_term
```

Transfer time = **latency + bytes / bandwidth**. This project mainly uses a fixed latency term; set `bandwidth_bps > 0` to add a per-byte transfer cost.

### `DEFAULT_LINK`

A module-level default: `LinkProfile(latency_s=0.02)` — **20 ms** per transfer, deliberately exaggerated so the teaching effect is obvious. Pass a custom `LinkProfile` to tighten or loosen the bottleneck.

### `slow_all_to_all_single(output, input, link=DEFAULT_LINK)`

**Synchronous** All-to-All with simulated latency:

1. Sleep for `link.transfer_time(nbytes)` on the input tensor.
2. Call `dist.all_to_all_single(output, input)`.

Semantically equivalent to a blocking All-to-All; **naive_moe** uses this so every communication dutifully waits out the full latency while compute units idle.

### `slow_all_to_all_single_async(output, input, link=DEFAULT_LINK)`

**Asynchronous** All-to-All; returns a `_DelayedWork` handle immediately:

1. Fire `dist.all_to_all_single(..., async_op=True)` right away.
2. Book the modeled latency (`link.transfer_time(nbytes)`) onto the handle.
3. Caller calls `.wait()` when the output buffer is actually needed.

**lightning_moe** uses this to pipeline tiles: compute on tile *t* while tile *t+1*'s All-to-All is in flight.

## How to use it

Import from `topology_sim` inside the MoE demos (the demos add the parent directory to `sys.path`):

```python
from topology_sim import DEFAULT_LINK, slow_all_to_all_single, slow_all_to_all_single_async
```

### Synchronous (naive MoE)

```python
dispatched = torch.empty_like(tokens)
with timeline.span(rank, "dispatch", kind="comm"):
    slow_all_to_all_single(dispatched, tokens, DEFAULT_LINK)
# dispatched now holds tokens destined for this rank's expert
out = expert(dispatched)
```

### Asynchronous (lightning MoE)

```python
buf0 = torch.empty_like(tile0)
work0 = slow_all_to_all_single_async(buf0, tile0, DEFAULT_LINK)

# ... fire next tile's async dispatch, then do expert compute on the device ...

with timeline.span(rank, "dispatch_wait", kind="comm"):
    work0.wait()
```

Quick self-check (single process, no rendezvous):

```bash
python 4_expert_parallel/topology_sim.py
```

Prints injected delay ≈ 20 ms.

## How the overlap is simulated

Truly modeling both **hardware DMA overlap** and **network latency** on a Mac is not trivial. This teaching implementation uses a deliberate compromise:

| Aspect | What is real | What is simulated |
|--------|--------------|-------------------|
| Non-blocking collective | `async_op=True` fires immediately | — |
| Cross-machine latency | — | `time.sleep` booked on the handle |
| Overlap shape | Compute before `.wait()` runs concurrently with in-flight comm | — |
| Overlap amount | Elapsed time since handle creation subtracted at `.wait()` | Remaining latency topped up with sleep |

**`_DelayedWork`** records `t0` when the async All-to-All starts. On `.wait()`:

1. `elapsed = now - t0`
2. `remaining = delay_s - elapsed`
3. If `remaining > 0`, sleep that long (latency not yet "eaten" by intervening compute)
4. Call `self._work.wait()` on the real dist work handle

If you fire tile *t+1*'s dispatch and then spend time in `expert(buf_t)` on the device, that compute time reduces the sleep at `.wait()` — exactly the comm-compute overlap lightning MoE targets. The **shape** of overlap is real; the **amount** of latency is simulated.

## Papers & further reading

- Hockney, 1994, "The communication challenge for MPP: Intel Paragon and Meiko CS-2" (alpha-beta latency-bandwidth model).
- Thakur, Rabenseifner & Gropp, 2005, "Optimization of Collective Communication Operations in MPICH".
