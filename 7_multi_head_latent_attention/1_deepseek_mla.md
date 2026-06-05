# MLA: Multi-head Latent Attention (Low-Rank KV Compression)

> Compress the KV cache into a small latent vector per token, reconstruct K and V on the fly, and shard heads across ranks with All-Gather to stitch the full attention output.

## TL;DR

- Standard MHA caches full $K$ and $V$ per token, each of size $H \cdot D_h$ — this dominates inference memory at long context.
- MLA (DeepSeek-V2) **down-projects** the hidden state to a latent $c_{KV} \in \mathbb{R}^{d_c}$ with $d_c \ll H \cdot D_h$; only $c_{KV}$ is cached.
- $K$ and $V$ are **up-projected** from the latent when needed; cache shrinks by factor $\approx (H \cdot D_h) / d_c$ with negligible quality loss.
- This demo also **tensor-parallelizes over heads** (Module 2 style): each rank up-projects and attends its head group, then **All-Gather** concatenates per-head outputs into the full $[T, H \cdot D_h]$ result.
- The shared latent $c_{KV}$ is cheap — every rank recomputes it locally; no communication in the projection step.

## The problem

During autoregressive inference, the KV cache grows linearly with sequence length. For each new token you must store a full key vector and a full value vector per head. With $H$ heads and head dimension $D_h$, that is $2 \cdot H \cdot D_h$ floats per token — often the largest resident memory in a long-context deployment. Tensor and pipeline parallelism help fit the model, but they do not shrink the per-token cache footprint. MLA attacks the cache directly: instead of storing $K$ and $V$, store a single low-rank latent that captures the information needed to reconstruct them.

## Algorithm

Let hidden states be $x \in \mathbb{R}^{T \times d_{\text{model}}}$, number of heads $H$, head dim $D_h$, latent dim $d_c$, and scale $= 1/\sqrt{D_h}$. This demo uses `SEQ=12`, `D_MODEL=32`, `HEADS=4`, `HEAD_DIM=8`, `LATENT_DIM=16`, and `WORLD_SIZE=4` (one head per rank).

1. **Down-project to the latent (the cache)**:
   $$c_{KV} = x \, W_{DKV} \in \mathbb{R}^{T \times d_c}, \quad W_{DKV} \in \mathbb{R}^{d_{\text{model}} \times d_c}.$$
   At inference, only $c_{KV}$ is appended to the cache — not full $K$ or $V$.

2. **Up-project latent to full K and V** (reconstructed on demand):
   $$K = c_{KV} \, W_{UK} \in \mathbb{R}^{T \times H D_h}, \quad W_{UK} \in \mathbb{R}^{d_c \times H D_h},$$
   $$V = c_{KV} \, W_{UV} \in \mathbb{R}^{T \times H D_h}, \quad W_{UV} \in \mathbb{R}^{d_c \times H D_h}.$$
   Reshape to $K, V \in \mathbb{R}^{T \times H \times D_h}$ for per-head attention.

3. **Query projection** (not compressed in this teaching demo):
   $$Q = x \, W_Q \in \mathbb{R}^{T \times H D_h}, \quad W_Q \in \mathbb{R}^{d_{\text{model}} \times H D_h}.$$
   Reshape to $Q \in \mathbb{R}^{T \times H \times D_h}$.

4. **Per-head scaled dot-product attention**:
   $$\text{Attn}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{D_h}}\right) V_h \in \mathbb{R}^{T \times D_h}, \quad h = 0, \ldots, H-1.$$
   Concatenate head outputs: $\text{out} \in \mathbb{R}^{T \times H D_h}$.

5. **Tensor parallel over heads** (this demo): rank $r$ owns head group $h \in [r \cdot H/n,\; (r+1) \cdot H/n)$ via column slices of $W_{UK}$, $W_{UV}$, and $W_Q$. Each rank computes local $\text{Attn}_r \in \mathbb{R}^{T \times (H/n) D_h}$, then All-Gather + concat yields the full output.

6. **Production note (not implemented here)**: DeepSeek-V2's "absorption" trick folds $W_{UK}$ into $W_Q$ and $W_{UV}$ into the output projection so $K$ and $V$ need never be materialized at inference. The full method also uses decoupled RoPE keys; this teaching demo omits RoPE for clarity.

With `HEADS * HEAD_DIM = 32` and `LATENT_DIM = 16`, the KV cache shrinks by a factor of **~2×** per token (in general, $\approx (H D_h) / d_c$).

## Communication pattern

```mermaid
sequenceDiagram
    participant R0 as Rank 0 (head 0)
    participant R1 as Rank 1 (head 1)
    participant R2 as Rank 2 (head 2)
    participant R3 as Rank 3 (head 3)

    Note over R0,R3: Input x replicated on every rank (T × d_model)

    Note over R0,R3: Local down-project (no comm) — same c_KV on all ranks
    R0->>R0: c_KV = x @ W_DKV  (T × d_c)
    R1->>R1: c_KV = x @ W_DKV  (T × d_c)
    R2->>R2: c_KV = x @ W_DKV  (T × d_c)
    R3->>R3: c_KV = x @ W_DKV  (T × d_c)

    Note over R0: up-project head 0 slice
    Note over R1: up-project head 1 slice
    Note over R2: up-project head 2 slice
    Note over R3: up-project head 3 slice

    R0->>R0: q₀,k₀,v₀ from c_KV & x; local Attn₀
    R1->>R1: q₁,k₁,v₁ from c_KV & x; local Attn₁
    R2->>R2: q₂,k₂,v₂ from c_KV & x; local Attn₂
    R3->>R3: q₃,k₃,v₃ from c_KV & x; local Attn₃

    Note over R0,R3: All-Gather on COMM_DEVICE (CPU)
    R0->>R0: gather [Attn₀, Attn₁, Attn₂, Attn₃]
    R1->>R1: gather [Attn₀, Attn₁, Attn₂, Attn₃]
    R2->>R2: gather [Attn₀, Attn₁, Attn₂, Attn₃]
    R3->>R3: gather [Attn₀, Attn₁, Attn₂, Attn₃]

    Note over R0,R3: cat(dim=-1) → full out (T × H·D_h) on each rank
```

## What you'll see

On launch, `launch_teaching_cluster` prints a banner with `world_size=4` and `backend=gloo`, then spawns four color-coded rank logs. Each rank reports the cache compression ratio, e.g.:

```
latent dim d_c = 16 vs full KV per token = 32 (cache shrinks ~2.0x)
```

Before your implementation, the run stops at:

```
NotImplementedError: TODO: compress_and_project -- hand-write MLA latent down/up projection
```

After a correct implementation, rank 0 prints something like:

```
MLA output shape = (12, 32) | max error vs single-machine = 1.19e-05
```

The demo builds identical weights on every rank, computes a single-machine MLA reference on CPU, and reports the max absolute error against your distributed result. A correct down/up projection plus All-Gather should land near **~1e-5** (float32 noise).

## Your battle zone

Implement **`compress_and_project(x, w, rank, device)`** in `7_multi_head_latent_attention/deepseek_mla.py`. The skeleton raises `NotImplementedError`; you fill in:

1. **Down-project to latent**: `c_kv = x @ w["W_DKV"]` → shape `[SEQ, LATENT_DIM]`.
2. **Up-project K for this rank's head slice**: `k = (c_kv @ w["W_UK"][:, sl]).view(SEQ, HEADS_PER_RANK, HEAD_DIM)` where `sl = head_slice(rank)`.
3. **Up-project V for this rank's head slice**: `v = (c_kv @ w["W_UV"][:, sl]).view(SEQ, HEADS_PER_RANK, HEAD_DIM)`.
4. **Project Q for this rank's head slice**: `q = (x @ w["W_Q"][:, sl]).view(SEQ, HEADS_PER_RANK, HEAD_DIM)`.
5. **Return** `(q, k, v)`.

No communication in this function — it is local compute on the MPS `device`. Helpers **`local_attention`** (per-head softmax attention) and **`gather_heads`** (All-Gather on **`COMM_DEVICE`**, then concat) are already provided.

Remember the golden rule: compute on MPS, move to **`COMM_DEVICE`** before gloo collectives, move back after.

## Run it

```bash
python 7_multi_head_latent_attention/deepseek_mla.py
```

## Papers & further reading

- DeepSeek-AI, 2024, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (Multi-head Latent Attention).
