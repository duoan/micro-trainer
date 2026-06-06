# Module 3 — Pipeline Parallelism (scheduling state machines)

> Split the model by depth into stages, stream micro-batches through them, and fight the "bubble" with ever-smarter schedules.

These are the hardest demos in the repo: hand-written warmup/steady/cooldown state machines over raw `send`/`recv`. Get GPipe and 1F1B solid before the advanced schedules.

| # | Demo | Doc | Difficulty | What you hand-write |
| - | ---- | --- | ---------- | ------------------- |
| 1 | [`1_gpipe.py`](1_gpipe.py) | [doc](1_gpipe.md) | ★★★☆☆ | F-then-B schedule over raw `send`/`recv` |
| 2 | [`2_one_forward_backward.py`](2_one_forward_backward.py) | [doc](2_one_forward_backward.md) | ★★★★☆ | the 1F1B warmup/steady/cooldown machine |
| 3 | [`3_interleaved_1f1b.py`](3_interleaved_1f1b.py) | [doc](3_interleaved_1f1b.md) | ★★★★★ | virtual pipeline: V chunks per rank (~1/V bubble) |
| 4 | [`4_pipedream.py`](4_pipedream.py) | [doc](4_pipedream.md) | ★★★★☆ | async schedule + weight stashing |
| 5 | [`5_chimera.py`](5_chimera.py) | [doc](5_chimera.md) | ★★★★★ | two opposing pipelines (halved bubble) |
| 6 | [`6_deepseek_dualpipe.py`](6_deepseek_dualpipe.py) | [doc](6_deepseek_dualpipe.md) | ★★★★★ | [DeepSeek] bidirectional pipeline + comm/compute overlap |

Run any demo directly (from the repo root, no env vars):

```bash
python 3_pipeline_parallel/1_gpipe.py
```

← Back to the [project README](../README.md) for setup and the full cross-module difficulty ladder.
