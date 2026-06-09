# Distributed Data Parallel (`1_ddp.py`)

> One `MicroDDP` wrapper, three levels that compute the **same** averaged gradient, each faster than the last: `naive` → `overlap` → `bucketing`.

Run a single level:

```bash
python 1_data_parallel/1_ddp.py naive      # or: overlap / bucketing
```

## The idea

Data parallelism: every rank trains on a different slice of the global batch, so each ends up with gradients from its own shard only. Because all ranks share the same weights, every step must **average** the gradients before the optimizer step — that is what makes it equivalent to training on one big batch. Averaging = `All-Reduce(SUM) / world_size`. The three levels differ only in **how** that reduction is issued.

`MicroDDP(nn.Module)` broadcasts the weights on construction so every rank starts identical, delegates `forward`, and exposes `sync_grads()` to call after `backward()`. A `syncing` flag keeps the hook-based levels dormant while the demo computes a clean full-batch reference.

## The three levels

| Level | Mechanism | Battle zone |
| ----- | --------- | ----------- |
| `naive` | one `all_reduce` **per parameter**, after backward finishes | `_sync_naive` |
| `overlap` | each grad's `all_reduce` fired **async from a backward hook**, so comm overlaps with the rest of backward | `_overlap_hook` |
| `bucketing` | coalesce a whole **bucket** of grads into one buffer and reduce it once (amortize per-collective cost) | `_reduce_bucket` |

- **naive** — for each `p.grad`: `dist.all_reduce(p.grad, op=SUM)` then `/= world_size`. The "hello world" of collectives.
- **overlap** — a parameter's grad is ready the moment its layer's backward finishes. In the hook: scale by `1/world_size`, fire a non-blocking `all_reduce(async_op=True)`, and stash the handle; `sync_grads` waits on all handles. Communication hides behind the rest of backward (the provided plumbing waits; you write the launch).
- **bucketing** — grads become ready last-layer-first, so params are bucketed in reverse order (provided). When a bucket's grads are all in, you flatten them into one buffer, scale, fire **one** async `all_reduce`, and record it so `sync_grads` can scatter the averaged result back into each grad.

## What you'll see

On a fresh clone each level stops at its `NotImplementedError`. Once implemented, rank 0 prints `[level] max err vs full-batch grad = …e-07` — all three produce the same average, confirmed against a single-process full-batch reference.

> On a single CPU box there's no wall-clock win to observe — the lesson is the **mechanism** (per-param vs hook/async vs bucketed). The exact same code is what pays off on a real GPU cluster.

## Papers & further reading

- Li et al., 2020, "PyTorch Distributed: Experiences on Accelerating Data Parallel Training" (VLDB) — gradient bucketing + overlap with backward.
- PyTorch docs: `Tensor.register_post_accumulate_grad_hook`, `torch.distributed` async collectives (`async_op=True`, `Work.wait()`).
