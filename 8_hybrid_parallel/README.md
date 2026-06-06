# Module 8 — Hybrid / N-D Mesh Parallelism (the capstone)

> The grand finale: compose the axes you built in modules 1-7 onto a single device mesh. Save this for last.

| # | Demo | Doc | Difficulty | What you hand-write |
| - | ---- | --- | ---------- | ------------------- |
| 1 | [`1_three_d_parallel.py`](1_three_d_parallel.py) | [doc](1_three_d_parallel.md) | ★★★★★ | DP × TP × PP: TP All-Reduce in a layer, PP send/recv across stages, DP All-Reduce across replicas |
| 2 | [`2_nd_mesh_parallel.py`](2_nd_mesh_parallel.py) | [doc](2_nd_mesh_parallel.md) | ★★★★★ | full **6D** mesh `[TP, EP, CP, FS, DP, PP]` + per-axis group "flash" |

`2_nd_mesh_parallel.py` is a terminal take on [Visualizing 6D Mesh Parallelism](https://main-horse.github.io/posts/visualizing-6d/); its doc doubles as a **0D → 6D** map linking every dimension back to the module that hand-writes it.

Run any demo directly (from the repo root, no env vars):

```bash
python 8_hybrid_parallel/1_three_d_parallel.py
python 8_hybrid_parallel/2_nd_mesh_parallel.py
```

← Back to the [project README](../README.md) for setup and the full cross-module difficulty ladder.
