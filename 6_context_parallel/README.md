# Module 6 — Context Parallelism

> Attention over a near-infinite context: shard the sequence across devices and rotate K/V blocks around a ring, merging partial attentions with online softmax.

| # | Demo | Doc | Difficulty | What you hand-write |
| - | ---- | --- | ---------- | ------------------- |
| 1 | [`1_ring_attention.py`](1_ring_attention.py) | [doc](1_ring_attention.md) | ★★★★☆ | online softmax + K/V ring rotation (P2P) |

Run it directly (from the repo root, no env vars):

```bash
python 6_context_parallel/1_ring_attention.py
```

← Back to the [project README](../README.md) for setup and the full cross-module difficulty ladder.
