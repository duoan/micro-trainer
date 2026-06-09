# Fully Sharded Data Parallel — the ZeRO ladder (`2_fsdp.py`)

> One `MicroFSDP` wrapper, three escalating stages that shard ever more of the training state: `zero1` (optimizer state) → `zero2` (+ gradients) → `zero3` (+ parameters).

Run a single stage:

```bash
python 1_data_parallel/2_fsdp.py zero1     # or: zero2 / zero3
```

## The idea

In naive DDP every rank keeps a full copy of the parameters, gradients, **and** optimizer state — lots of duplication. ZeRO removes that duplication in stages. We use one unifying trick: flatten all parameters into a single vector and give **rank `r` ownership of chunk `r`**. Each rank builds its optimizer over only its chunk, so **optimizer state is sharded in all three stages**. What escalates is what *else* gets sharded:

| Stage | Also shards | Grad reduction | Params at rest |
| ----- | ----------- | -------------- | -------------- |
| `zero1` | — (optimizer state only) | `all_reduce` full grad, keep my chunk | full (replicated) |
| `zero2` | gradients | `reduce_scatter` → only my chunk's avg grad | full (replicated) |
| `zero3` | gradients **and** parameters | `reduce_scatter` | **only my chunk** (gathered on demand) |

## The collectives (your battle zones)

Each stage's signature collective is a battle zone; they're cumulative:

- **`_all_reduce_grad`** (zero1) — `all_reduce(SUM)` the full gradient, `/ world_size`, return `chunk[rank]`. Wasteful (every rank moves the whole gradient) — which is exactly what zero2 fixes.
- **`_reduce_scatter_grad`** (zero2, zero3) — `reduce_scatter(SUM)` the per-chunk gradients so rank `r` receives only its chunk's sum, `/ world_size`. No rank ever holds a gradient slice it doesn't own.
- **`gather_params`** (all stages) — `all_gather` every rank's chunk, concatenate into the full flat vector, and load it back into the model. This is FSDP's "materialize parameters from shards" step.

The per-step orchestration (provided in `run`) is where zero3 differs: it `gather_params()` **before** the forward (materialize from shards) and drops the full copy after the step, so the full parameters never all live in memory at once. zero1/zero2 keep params resident and `gather_params()` **after** the step to rebuild them for the next forward.

## What you'll see

Each stage prints its shard size, e.g. `[zero2] my shard = … elems (1/4 of … params)`, and a per-step `grad-chunk err` against the global-average gradient. When all the stage's battle zones are filled, rank 0 prints a cross-rank parameter drift of `0.00e+00` — every rank stays in sync.

To clear the dashboard: `zero1` needs `_all_reduce_grad` + `gather_params`; `zero2` needs `_reduce_scatter_grad` + `gather_params`; `zero3` needs the same two (its extra saving is the param lifecycle, shown in the provided loop).

## Papers & further reading

- Rajbhandari et al., 2020, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (SC20) — stages $P_{os}$, $P_{os+g}$, $P_{os+g+p}$.
- Zhao et al., 2023, "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel".
- PyTorch docs: `torch.distributed.all_reduce` / `reduce_scatter` / `all_gather`.
