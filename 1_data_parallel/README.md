# Module 1 — Data Parallelism & Memory Sharding

> Replicate the model, shard the data; then progressively shard the *state* (optimizer → gradients → parameters) to fit bigger models.

Three wrappers, each with selectable **levels** (so the boilerplate lives in one place — see [`_common.py`](_common.py)). First make DDP itself faster, then climb the "memory sharding ladder" (ZeRO stages), then go 2D with HSDP.

| # | Demo | Doc | Levels | What you hand-write |
| - | ---- | --- | ------ | ------------------- |
| 1 | [`1_ddp.py`](1_ddp.py) | [doc](1_ddp.md) | `naive` / `overlap` / `bucketing` | `MicroDDP`: per-param `all_reduce` → async from backward hooks → bucketed |
| 2 | [`2_fsdp.py`](2_fsdp.py) | [doc](2_fsdp.md) | `zero1` / `zero2` / `zero3` | `MicroFSDP`: shard optimizer state → + grads (`reduce_scatter`) → + params (`all_gather`) |
| 3 | [`3_hsdp.py`](3_hsdp.py) | [doc](3_hsdp.md) | `hsdp` | `MicroHSDP`: 2D mesh — intra-node FSDP `reduce_scatter` + inter-node DDP `all_reduce` (`new_group`) |

Run a single level (from the repo root, no env vars):

```bash
python 1_data_parallel/1_ddp.py overlap      # 2_fsdp.py zero2, 3_hsdp.py, ...
```

← Back to the [project README](../README.md) for setup and the full cross-module difficulty ladder.
