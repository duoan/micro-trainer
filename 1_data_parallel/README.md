# Module 1 — Data Parallelism & Memory Sharding

> Replicate the model, shard the data; then progressively shard the *state* (optimizer → gradients → parameters) to fit bigger models.

First make DDP itself faster (naive → bucketing → overlap), then climb the "memory sharding ladder" (ZeRO stages). Do them in order — each one removes a cost or redundancy the previous one still kept. Demos 1–3 build a hand-written `MicroDDP(nn.Module)` one battle zone at a time.

| # | Demo | Doc | Difficulty | What you hand-write |
| - | ---- | --- | ---------- | ------------------- |
| 1 | [`1_ddp_demo.py`](1_ddp_demo.py) | [doc](1_ddp_demo.md) | ★☆☆☆☆ | gradient `all_reduce` averaging (the "hello world") |
| 2 | [`2_ddp_bucketing.py`](2_ddp_bucketing.py) | [doc](2_ddp_bucketing.md) | ★★☆☆☆ | fuse all grads into ONE `all_reduce` (bucketing) |
| 3 | [`3_ddp_overlap.py`](3_ddp_overlap.py) | [doc](3_ddp_overlap.md) | ★★★☆☆ | async `all_reduce` from backward hooks (overlap comm/compute) |
| 4 | [`4_zero1_demo.py`](4_zero1_demo.py) | [doc](4_zero1_demo.md) | ★★☆☆☆ | shard optimizer state: ownership + `broadcast` |
| 5 | [`5_fsdp_demo.py`](5_fsdp_demo.py) | [doc](5_fsdp_demo.md) | ★★☆☆☆ | shard params & grads: `all_gather` + `reduce_scatter` |
| 6 | [`6_hsdp_demo.py`](6_hsdp_demo.py) | [doc](6_hsdp_demo.md) | ★★★☆☆ | 2D mesh: intra-node FSDP + inter-node DDP (`new_group`) |

Run any demo directly (from the repo root, no env vars):

```bash
python 1_data_parallel/1_ddp_demo.py
```

← Back to the [project README](../README.md) for setup and the full cross-module difficulty ladder.
