# Module 7 — Multi-head Latent Attention (MLA)

> DeepSeek-V2's attention variant: cache a small low-rank *latent* instead of the full K/V, shrinking the KV cache dramatically while keeping multi-head expressivity.

| # | Demo | Doc | Difficulty | What you hand-write |
| - | ---- | --- | ---------- | ------------------- |
| 1 | [`1_deepseek_mla.py`](1_deepseek_mla.py) | [doc](1_deepseek_mla.md) | ★★★☆☆ | low-rank KV compression + up-projection, TP over heads |

Run it directly (from the repo root, no env vars):

```bash
python 7_multi_head_latent_attention/1_deepseek_mla.py
```

← Back to the [project README](../README.md) for setup and the full cross-module difficulty ladder.
