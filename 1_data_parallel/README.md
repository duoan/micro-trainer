# Module 1 — Data Parallelism & Memory Sharding

> Replicate the model, shard the data; then progressively shard the *state* (optimizer → gradients → parameters) to fit bigger models.

This is the "memory sharding ladder": start from naive DDP, then climb the ZeRO stages. Do them in order — each one removes a redundancy the previous one still kept.

| # | Demo | Doc | Difficulty | What you hand-write |
| - | ---- | --- | ---------- | ------------------- |
| 1 | [`1_ddp_demo.py`](1_ddp_demo.py) | [doc](1_ddp_demo.md) | ★☆☆☆☆ | gradient `all_reduce` averaging (the "hello world") |
| 2 | [`2_zero1_demo.py`](2_zero1_demo.py) | [doc](2_zero1_demo.md) | ★★☆☆☆ | shard optimizer state: ownership + `broadcast` |
| 3 | [`3_fsdp_demo.py`](3_fsdp_demo.py) | [doc](3_fsdp_demo.md) | ★★☆☆☆ | shard params & grads: `all_gather` + `reduce_scatter` |
| 4 | [`4_hsdp_demo.py`](4_hsdp_demo.py) | [doc](4_hsdp_demo.md) | ★★★☆☆ | 2D mesh: intra-node FSDP + inter-node DDP (`new_group`) |

Run any demo directly (from the repo root, no env vars):

```bash
python 1_data_parallel/1_ddp_demo.py
```

← Back to the [project README](../README.md) for setup and the full cross-module difficulty ladder.
