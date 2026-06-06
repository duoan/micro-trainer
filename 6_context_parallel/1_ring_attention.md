# Context Parallel: Ring Attention with Online Softmax

> Split Q, K, V across ranks by sequence, rotate K/V blocks around a ring, and accumulate exact attention for local queries without ever materializing the full [S, S] matrix.

## TL;DR

- When sequence length $S$ makes even one attention matrix $\in \mathbb{R}^{S \times S}$ too large, **Context Parallel (CP)** shards Q, K, V along the sequence: rank $r$ holds chunks of length $L = S/n$.
- Every local query must attend to **all** keys and values. A **ring** rotates $(K, V)$ blocks: send to rank $+1$, receive from rank $-1$, repeat $n$ steps.
- **Online softmax** (FlashAttention-style) maintains running row-max $m$, denominator `denom`, and weighted output `acc`, rescaling as each block arrives — no full softmax over $S$ keys at once.
- This demo computes **non-causal full attention** (every block contributes; no mask). `scale = 1/sqrt(D)`.

## The problem

Attention is $O(S^2)$ in memory for the score matrix. Sequence Parallel and Tensor Parallel shrink per-device activations in linear layers, but a single attention head still needs every query row to dot-product with every key column. Context Parallelism (Ring Attention) splits the sequence across ranks so each holds $Q_r, K_r, V_r \in \mathbb{R}^{L \times D}$ with $L = S/n$. The fix: keep $Q_r$ fixed on rank $r$ and circulate each rank's $(K, V)$ block around the ring over $n$ hops. After $n$ steps, rank $r$ has seen every K/V block and can produce the exact attention output for its local queries — without storing $\mathbb{R}^{S \times S}$ anywhere.

## Algorithm

Let `world_size` $= n$, full sequence $S$, local chunk $L = S/n$, head dim $D$, scale $= 1/\sqrt{D}$. Rank $r$ holds $Q_r, K_r^{(0)}, V_r^{(0)} \in \mathbb{R}^{L \times D}$ (initial block is its own shard). The demo uses `SEQ=16`, `DIM=8`, `WORLD_SIZE=4`, so $L=4$.

1. **Initialize running stats** (your TODO):
   $$m = -\infty \in \mathbb{R}^{L \times 1}, \quad \text{denom} = 0 \in \mathbb{R}^{L \times 1}, \quad \text{acc} = 0 \in \mathbb{R}^{L \times D}.$$
2. **Ring loop** — repeat $n$ times (one per K/V block):
   1. Compute block scores:
      $$\text{scores} = (Q_r \; K_{\text{blk}}^\top) \cdot \text{scale} \in \mathbb{R}^{L \times L}.$$
   2. **Online softmax update** (`online_softmax_update`, provided):
      - $\text{block\_max} = \text{rowmax}(\text{scores})$
      - $m_{\text{new}} = \max(m, \text{block\_max})$
      - $\text{correction} = \exp(m - m_{\text{new}})$
      - $p = \exp(\text{scores} - m_{\text{new}})$
      - $\text{denom} \leftarrow \text{denom} \cdot \text{correction} + \text{rowsum}(p)$
      - $\text{acc} \leftarrow \text{acc} \cdot \text{correction} + p \; V_{\text{blk}}$
   3. If not the last step, **ring exchange** (`ring_exchange`, provided): pack `kv = cat([k_blk, v_blk], dim=-1)`, send to rank $+1$ / recv from rank $-1$ via non-blocking `dist.isend`/`dist.irecv` on the same `device`, split received buffer back into $K_{\text{blk}}, V_{\text{blk}}$.
3. **Normalize**: $\text{out}_r = \text{acc} / \text{denom} \in \mathbb{R}^{L \times D}$.

After the loop, rank $r$ holds the exact softmax attention output for its local query rows against all $S$ keys.

## Communication pattern

```mermaid
sequenceDiagram
    participant R0 as Rank 0
    participant R1 as Rank 1
    participant R2 as Rank 2
    participant R3 as Rank 3

    Note over R0,R3: Each rank keeps Q fixed (L × D); K/V rotate

    Note over R0,R3: Step 0 — local (K, V) block
    R0->>R0: scores₀ = Q₀ @ K₀ᵀ, online_softmax_update
    R1->>R1: scores₁ = Q₁ @ K₁ᵀ, online_softmax_update
    R2->>R2: scores₂ = Q₂ @ K₂ᵀ, online_softmax_update
    R3->>R3: scores₃ = Q₃ @ K₃ᵀ, online_softmax_update

    Note over R0,R3: Step 1 — ring_exchange (isend→r+1, irecv←r−1)
    R0->>R1: send K₀|V₀
    R1->>R2: send K₁|V₁
    R2->>R3: send K₂|V₂
    R3->>R0: send K₃|V₃
    R0->>R0: recv K₃|V₃, update
    R1->>R1: recv K₀|V₀, update
    R2->>R2: recv K₁|V₁, update
    R3->>R3: recv K₂|V₂, update

    Note over R0,R3: Steps 2–3 — repeat until all n blocks visited
    Note over R0,R3: out = acc / denom on each rank (L × D)
```

## What you'll see

On launch, `launch_teaching_cluster` prints a banner with `world_size=4` and `backend=gloo`, then spawns four color-coded rank logs. Each rank reports its local chunk, e.g. `local Q/K/V chunk = (4, 8) (full seq = 16)`. Before your implementation, the run stops at:

```
NotImplementedError: TODO: ring_attention -- hand-write the ring loop + online softmax accumulation
```

After a correct implementation, rank 0 prints something like:

```
Ring-Attention output shape = (16, 8) | max error vs single-machine = 1.19e-06
```

The demo builds full Q, K, V via `build_full_qkv`, computes single-machine reference $\text{softmax}(Q K^\top / \sqrt{D}) V$, gathers local outputs with All-Gather, and reports max absolute error. A correct ring + online softmax should match within **~1e-6**.

## Your battle zone

Implement **`ring_attention(q_local, k_local, v_local, rank, world_size, device)`** in `6_context_parallel/1_ring_attention.py`. The skeleton raises `NotImplementedError`; you fill in:

1. Init $m = \text{full}([L, 1], -\infty)$, `denom = zeros([L, 1])`, `acc = zeros([L, DIM])`.
2. Start with `k_blk, v_blk = k_local, v_local`.
3. Loop `world_size` times:
   - `scores = (q_local @ k_blk.transpose(0, 1)) * SCALE`
   - `m, denom, acc = online_softmax_update(m, denom, acc, scores, v_blk)`
   - If not the last step: pack `kv = torch.cat([k_blk, v_blk], dim=-1)`, call `ring_exchange(kv, rank, world_size, device)`, split into `k_blk, v_blk`.
4. Return `acc / denom`.

Helpers `ring_exchange` and `online_softmax_update` are provided. Everything — compute and ring send/recv — runs on the same `device` (gloo on CPU, NCCL on GPU).

## Run it

```bash
python 6_context_parallel/1_ring_attention.py
```

## Papers & further reading

- Liu, Zaharia & Abbeel, 2023, "Ring Attention with Blockwise Transformers for Near-Infinite Context".
- Dao et al., 2022, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness".
- Milakov & Gimelshein, 2018, "Online normalizer calculation for softmax".
