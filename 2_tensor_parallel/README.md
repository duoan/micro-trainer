# Module 2 — Tensor Parallelism (Megatron matrix sharding)

> Split a single layer's weight matrix across devices, so one matmul becomes many partial matmuls plus one collective.

Start with Megatron's 1D split (column then row), then see how a 2D grid (SUMMA) cuts the communication volume at scale.

| # | Demo | Doc | Difficulty | What you hand-write |
| - | ---- | --- | ---------- | ------------------- |
| 1 | [`1_column_parallel.py`](1_column_parallel.py) | [doc](1_column_parallel.md) | ★☆☆☆☆ | column split: local matmul + `all_gather` concat |
| 2 | [`2_row_parallel.py`](2_row_parallel.py) | [doc](2_row_parallel.md) | ★☆☆☆☆ | row split: partial sums + `all_reduce` |
| 3 | [`3_summa_2d.py`](3_summa_2d.py) | [doc](3_summa_2d.md) | ★★★☆☆ | 2D/2.5D SUMMA: row/column `broadcast` on a √N grid |

Run any demo directly (from the repo root, no env vars):

```bash
python 2_tensor_parallel/1_column_parallel.py
```

← Back to the [project README](../README.md) for setup and the full cross-module difficulty ladder.
