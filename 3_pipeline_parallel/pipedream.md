# PipeDream: Asynchronous Pipeline + Weight Stashing

> Never flush the pipeline — update each stage as soon as a backward arrives. Keep correctness by stashing the exact weight version each micro-batch's forward used, and restoring it for that micro-batch's backward.

## TL;DR

- GPipe/1F1B are **synchronous**: flush and apply one global gradient before the next batch → a bubble every iteration.
- PipeDream runs **asynchronously**: each stage steps its weights per micro-batch, keeping the pipe always full.
- The hazard: micro-batch $m$'s backward may run against newer weights than its forward used → wrong gradient.
- The fix: **weight stashing** — snapshot weights at forward, restore them for the matching backward.

## The problem

A synchronous pipeline wastes time draining and refilling around every optimizer step. PipeDream keeps every stage busy by never draining — but asynchronous updates break the forward/backward weight-consistency that SGD assumes. Without care, stage $i$ computes $\partial L/\partial w$ at weight version $v+2$ for a forward that ran at version $v$. PipeDream restores consistency cheaply with **weight stashing**.

## Algorithm

Steady-state 1F1B comm pattern (forward $r\to r+1$, backward $r\leftarrow r+1$), but **no global flush**. Each stage maintains:

- `version`: how many optimizer steps it has taken.
- `pending`: a queue of in-flight micro-batches, each storing its input, output, **stashed weights**, and the version used.

**forward(m)**: stash the current weights *before* compute, run `out = stage(x)`, push `{in, out, stash, ver}` to `pending`, send `out` downstream.

**backward(m)**: pop the oldest pending item, **restore its stashed weights**, compute the gradient against them, send the input-gradient upstream, then **restore the latest weights**, take an SGD step, and `version += 1`.

Because the backward uses the same weights the forward did, the async pipeline is numerically equivalent to a consistent per-micro-batch update — with a near-zero bubble.

## Communication pattern

```mermaid
sequenceDiagram
    participant S as Stage i
    Note over S: forward(m) at weight v
    S->>S: stash w_v ; out_m = f(x_m; w_v)
    Note over S: ... other micro-batches step weights to v+2 ...
    Note over S: backward(m) arrives
    S->>S: restore w_v → grad against w_v (correct!)
    S->>S: restore w_latest → SGD step → version++
```

## What you'll see

Each rank prints its warmup count and the async note. Before implementation:

```
NotImplementedError: TODO: pipedream_schedule -- hand-write async 1F1B + weight stashing
```

When implemented, the live log shows the weight **version** each F/B used — e.g. `[F5](w_v2)` then later `[B5](w_v2)` — making the "async but consistent" behavior visible, with a `render_gantt` timeline that has almost no flush bubble.

## Your battle zone

`pipedream_schedule(...)` in `3_pipeline_parallel/pipedream.py`: warmup / steady / cooldown with a `pending` queue, using the provided `stash_weights` / `restore_weights` helpers. Key invariant — **backward(m) restores `item["stash"]` before `out.backward(...)`**, then restores the latest weights before stepping.

## Run it

```bash
python 3_pipeline_parallel/pipedream.py
```

## Papers & further reading

- Narayanan et al., 2019, "PipeDream: Generalized Pipeline Parallelism for DNN Training" (SOSP19).
- Narayanan et al., 2021, "Memory-Efficient Pipeline-Parallel DNN Training" (PipeDream-2BW, ICML21).
- Harlap et al., 2018, "PipeDream: Fast and Efficient Pipeline Parallel DNN Training".
