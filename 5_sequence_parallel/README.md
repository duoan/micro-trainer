# Module 5 — Sequence Parallelism

> Shard activations along the sequence dimension to cut activation memory; two different communication recipes for the attention boundary.

| # | Demo | Doc | Difficulty | What you hand-write |
| - | ---- | --- | ---------- | ------------------- |
| 1 | [`1_megatron_sp.py`](1_megatron_sp.py) | [doc](1_megatron_sp.md) | ★★★☆☆ | the `all_gather` / `reduce_scatter` conjugate pair |
| 2 | [`2_ulysses_sp.py`](2_ulysses_sp.py) | [doc](2_ulysses_sp.md) | ★★★☆☆ | DeepSpeed-Ulysses `all_to_all` (swap sequence ↔ head) |

Run any demo directly (from the repo root, no env vars):

```bash
python 5_sequence_parallel/1_megatron_sp.py
```

← Back to the [project README](../README.md) for setup and the full cross-module difficulty ladder.
