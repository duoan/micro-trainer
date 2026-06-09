# Module 1 — Data Parallelism & Memory Sharding

> Replicate the model, shard the data; then progressively shard the *state* (optimizer → gradients → parameters) to fit bigger models.

First make DDP itself faster (naive → overlap → bucketing), then climb the "memory sharding ladder" (ZeRO stages). Do them in order — each one removes a cost or redundancy the previous one still kept. Demos 1–3 build a hand-written `MicroDDP(nn.Module)` one battle zone at a time.

| # | Demo | Doc | Difficulty | What you hand-write |
| - | ---- | --- | ---------- | ------------------- |
| 1 | [`1_ddp_demo.py`](1_ddp_demo.py) | [doc](1_ddp_demo.md) | ★☆☆☆☆ | gradient `all_reduce` averaging (the "hello world") |
| 2 | [`2_ddp_overlap.py`](2_ddp_overlap.py) | [doc](2_ddp_overlap.md) | ★★★☆☆ | async `all_reduce` from backward hooks (overlap comm/compute) |
| 3 | [`3_ddp_bucketing.py`](3_ddp_bucketing.py) | [doc](3_ddp_bucketing.md) | ★★★★☆ | reduce whole grad buckets async at once (the real DDP combo) |
| 4 | [`4_zero1_demo.py`](4_zero1_demo.py) | [doc](4_zero1_demo.md) | ★★☆☆☆ | ZeRO-1: shard optimizer state (`MicroZeroOptimizer`, `all_reduce` + `broadcast`) |
| 5 | [`5_zero2_demo.py`](5_zero2_demo.py) | [doc](5_zero2_demo.md) | ★★★☆☆ | ZeRO-2: one `MicroShardedDistributedDataParallel` wrapper shards grads (hook `reduce` to owner) + optimizer state |
| 6 | [`6_fsdp_demo.py`](6_fsdp_demo.py) | [doc](6_fsdp_demo.md) | ★★☆☆☆ | ZeRO-3 / FSDP: + shard params (`all_gather` + `reduce_scatter`) |
| 7 | [`7_hsdp_demo.py`](7_hsdp_demo.py) | [doc](7_hsdp_demo.md) | ★★★☆☆ | 2D mesh: intra-node FSDP + inter-node DDP (`new_group`) |

Run any demo directly (from the repo root, no env vars):

```bash
python 1_data_parallel/1_ddp_demo.py
```

← Back to the [project README](../README.md) for setup and the full cross-module difficulty ladder.
