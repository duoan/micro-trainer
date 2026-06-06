# Module 4 — Expert Parallelism (MoE)

> Spread experts across devices; route each token to its expert with All-to-All, then fight the cross-machine latency with overlap.

[`topology_sim.py`](topology_sim.py) is a **finished** helper (no battle zone): it injects an artificial cross-machine network delay so the comm/compute overlap of the later demos actually shows up on the Mac's fast unified memory.

| # | Demo | Doc | Difficulty | What you hand-write |
| - | ---- | --- | ---------- | ------------------- |
| — | [`topology_sim.py`](topology_sim.py) | [doc](topology_sim.md) | (helper, done) | artificial network bottleneck simulator |
| 1 | [`1_naive_moe.py`](1_naive_moe.py) | [doc](1_naive_moe.md) | ★★★★☆ | token routing + blocking `all_to_all` dispatch/combine |
| 2 | [`2_lightning_moe.py`](2_lightning_moe.py) | [doc](2_lightning_moe.md) | ★★★★☆ | tile-level comm/compute overlap |
| 3 | [`3_deepseek_moe.py`](3_deepseek_moe.py) | [doc](3_deepseek_moe.md) | ★★★★★ | [DeepSeek] fine-grained + shared experts, aux-loss-free balancing |
| 4 | [`4_deepseek_deepep.py`](4_deepseek_deepep.py) | [doc](4_deepseek_deepep.md) | ★★★★★ | [DeepSeek] overlapped 2-tier (NVLink/RDMA) `all_to_all` |

Run any demo directly (from the repo root, no env vars):

```bash
python 4_expert_parallel/1_naive_moe.py
```

← Back to the [project README](../README.md) for setup and the full cross-module difficulty ladder.
