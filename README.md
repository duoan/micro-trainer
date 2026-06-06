# micro-trainer

> Hand-write a complete large-model distributed training system, with multiple
> processes, on your Mac. No abstraction hell -- every `all_reduce` / `all_gather` /
> `send` / `recv` is laid out in the open for you to see.

`micro-trainer` is a hands-on, one-command-runnable teaching project for distributed
training. It needs no GPU cluster and no environment variables -- clone it, run
`python xxx_demo.py`, and it spins up a "pretend GPU cluster" with multiple processes. It
auto-adapts to your hardware (NCCL on a GPU box, gloo + MPS on an Apple-silicon Mac, gloo +
CPU otherwise), so you can see exactly how data / tensor / pipeline / sequence / context
parallelism and MoE communication overlap actually happen.

The foundation (multi-process rendezvous, device binding, port allocation, Timeline
dashboard base) is already built for you; **the hardest distributed operators and
scheduling logic are left for you to hand-write** -- every demo marks your "battle zone"
with `# TODO(you)`.

---

## How does one box simulate a cluster?

In real training, each process owns one GPU and processes talk to each other over
NCCL + an IB NIC. What we do here is essentially the same, we just swap "cards" for
"processes" and let `env_setup` **auto-detect your hardware**:

- **Processes as cards**: `mp.spawn` launches `world_size` child processes; each is one
  `rank` (~ one GPU).
- **Backend auto-detection**:
  - **NVIDIA GPU box** -> backend **NCCL**, and each rank binds its own `cuda:i`.
  - **Apple Silicon** -> backend **gloo** (a CPU comm backend), all ranks share one **MPS** device.
  - **Plain CPU** -> backend **gloo**, everything on `cpu`.
  - Force a choice any time with `MICRO_TRAINER_BACKEND=gloo|nccl`.

So the same `python xxx_demo.py` runs on a Mac laptop *and* on an 8-GPU server with no
code changes.

### One golden rule across the whole project: compute on `device`, communicate on `COMM_DEVICE`

`env_setup` exposes a `COMM_DEVICE` constant -- the device collectives must run on:

- **Under gloo (Mac/CPU)**: `COMM_DEVICE = cpu`, because **gloo cannot run collectives on
  MPS tensors**. So you compute on `device` (MPS) and `.to(COMM_DEVICE)` to move tensors to
  CPU before `dist.all_reduce` / `all_gather` / `send`, then move them back.
- **Under NCCL (GPU)**: `COMM_DEVICE` *is* the rank's CUDA device, so `.to(COMM_DEVICE)` is a
  **free no-op** -- NCCL talks GPU-to-GPU directly.

The beauty: **the exact same code** (`compute on device` -> `.to(COMM_DEVICE)` -> `communicate`
-> `.to(device)`) is correct on both backends. It also forces you to see that "compute" and
"communication" are two separate things -- the entire crux of distributed systems optimization.

---

## One-command run (Mac, GPU box, or CPU)

Prereq: install [uv](https://docs.astral.sh/uv/) (or any environment that can install PyTorch).

```bash
# Install dependencies (this installs PyTorch)
uv sync

# Run any demo directly -- no torchrun, no environment variables
uv run python 1_data_parallel/1_ddp_demo.py
```

> On the first run you'll see the cluster spin up and the data get sharded, then it stops
> at a `NotImplementedError` that says `TODO: ...` -- that is **intentional**! That is your
> battle zone. Fill in the `# TODO` and it runs through.

The launcher auto-detects your hardware: **NCCL + one `cuda:i` per rank on a GPU box**,
**gloo + shared MPS on Apple Silicon**, and **gloo + CPU otherwise**. The same command
works everywhere. Override with `MICRO_TRAINER_BACKEND=gloo` (e.g. to force the CPU path on
a GPU box). On a GPU box with fewer GPUs than ranks, ranks share GPUs round-robin -- fine
for small teaching runs.

---

## Project structure & learning path

Each module is a **"baseline -> evolution"** twin design: first write a naive version to
feel the pain, then write the evolved version that kills that pain.

```text
micro-trainer/
|-- env_setup/                  # foundation layer (already done, no edits needed)
|   `-- __init__.py             #   launch_teaching_cluster / bind_device / COMM_DEVICE / Timeline dashboard
|
|-- 1_data_parallel/            # Module 1: data parallelism & memory sharding
|   |-- 1_ddp_demo.py             #   baseline: naive DDP (hand-written gradient All-Reduce averaging)
|   |-- 2_zero1_demo.py           #   evolution: ZeRO-1 (optimizer state sharding + All-Gather)
|   |-- 3_fsdp_demo.py            #   evolution: ZeRO-3 / FSDP (param+grad sharding, All-Gather + Reduce-Scatter)
|   `-- 4_hsdp_demo.py            #   evolution: HSDP (2D mesh -- intra-node FSDP + inter-node DDP)
|
|-- 2_tensor_parallel/          # Module 2: Megatron matrix sharding
|   |-- 1_column_parallel.py      #   baseline: column split (All-Gather concat)
|   |-- 2_row_parallel.py         #   evolution: row split (All-Reduce sum)
|   `-- 3_summa_2d.py             #   evolution: 2D/2.5D SUMMA (sqrt(N) grid, row/col broadcasts)
|
|-- 3_pipeline_parallel/        # Module 3: pipeline scheduling state machine
|   |-- 1_gpipe.py                #   baseline: GPipe (F-then-B, big Bubble pain)
|   |-- 2_one_forward_backward.py #   evolution: 1F1B (one-forward-one-backward, live state machine)
|   |-- 3_interleaved_1f1b.py     #   evolution: Interleaved 1F1B (Megatron virtual pipeline, ~1/V bubble)
|   |-- 4_pipedream.py            #   evolution: PipeDream (async pipeline + weight stashing)
|   |-- 5_chimera.py              #   evolution: Chimera (bidirectional pipeline, halved bubble)
|   `-- 6_deepseek_dualpipe.py    #   [DeepSeek] DualPipe (bidirectional pipeline, near-zero bubble)
|
|-- 4_expert_parallel/          # Module 4: Expert Parallelism (MoE) -- All-to-All & comm overlap
|   |-- topology_sim.py           #   artificial network bottleneck simulator (injects cross-machine delay, done)
|   |-- 1_naive_moe.py            #   baseline: serial MoE (All-to-All dumb waiting)
|   |-- 2_lightning_moe.py        #   evolution: Lightning MoE (Tile-level micro-pipeline, full comm overlap)
|   |-- 3_deepseek_moe.py         #   [DeepSeek] DeepSeekMoE (fine-grained + shared experts, aux-loss-free balancing)
|   `-- 4_deepseek_deepep.py      #   [DeepSeek] DeepEP (overlapped dispatch + combine All-to-All, 2-tier net)
|
|-- 5_sequence_parallel/        # Module 5: Sequence Parallelism -- shard the sequence dimension
|   |-- 1_megatron_sp.py          #   baseline: Megatron SP (All-Gather / Reduce-Scatter conjugate pair)
|   `-- 2_ulysses_sp.py           #   evolution: DeepSpeed-Ulysses SP (All-to-All head/sequence swap)
|
|-- 6_context_parallel/         # Module 6: Context Parallelism -- attention over near-infinite context
|   `-- 1_ring_attention.py       #   Ring Attention (rotate K/V around a ring + online softmax)
|
|-- 7_multi_head_latent_attention/  # Module 7: efficient attention
|   `-- 1_deepseek_mla.py             #   [DeepSeek] MLA (low-rank KV-cache compression, TP over heads)
|
`-- 8_hybrid_parallel/          # Module 8: capstone -- compose the axes
    |-- 1_three_d_parallel.py     #   3D Parallelism (DP x TP x PP on one mesh)
    `-- 2_nd_mesh_parallel.py     #   N-D (6D) mesh (TP/EP/CP/FS/DP/PP), flash per-axis groups
```

**Per-module guides** — each folder has its own README:
[1 · Data Parallel](1_data_parallel/) ·
[2 · Tensor Parallel](2_tensor_parallel/) ·
[3 · Pipeline Parallel](3_pipeline_parallel/) ·
[4 · Expert Parallel](4_expert_parallel/) ·
[5 · Sequence Parallel](5_sequence_parallel/) ·
[6 · Context Parallel](6_context_parallel/) ·
[7 · Multi-head Latent Attention](7_multi_head_latent_attention/) ·
[8 · Hybrid / N-D Mesh](8_hybrid_parallel/)

### Learning path by difficulty (do NOT just go 1 → 8 in order!)

The module numbers group demos by *topic*, not by *difficulty*. A 1F1B pipeline state
machine (module 3) is far harder than a sharding All-Gather (module 1) or an attention
trick (module 7). Follow the **difficulty ladder** below instead -- each level only uses
ideas from the levels above it, so you always have the tools you need.

| Level | Difficulty | Demo | Doc | New skill you'll hand-write |
| ----- | ---------- | ---- | --- | --------------------------- |
| **0. Hello, collective** | ★☆☆☆☆ | [`1_data_parallel/1_ddp_demo.py`](1_data_parallel/1_ddp_demo.py) | [doc](1_data_parallel/1_ddp_demo.md) | a single `all_reduce` (gradient average) -- **start here** |
| | ★☆☆☆☆ | [`2_tensor_parallel/1_column_parallel.py`](2_tensor_parallel/1_column_parallel.py) | [doc](2_tensor_parallel/1_column_parallel.md) | `all_gather` + concat |
| | ★☆☆☆☆ | [`2_tensor_parallel/2_row_parallel.py`](2_tensor_parallel/2_row_parallel.py) | [doc](2_tensor_parallel/2_row_parallel.md) | `all_reduce` partial sums |
| **1. Sharding** | ★★☆☆☆ | [`1_data_parallel/2_zero1_demo.py`](1_data_parallel/2_zero1_demo.py) | [doc](1_data_parallel/2_zero1_demo.md) | ownership + `broadcast` (shard optimizer state) |
| | ★★☆☆☆ | [`1_data_parallel/3_fsdp_demo.py`](1_data_parallel/3_fsdp_demo.py) | [doc](1_data_parallel/3_fsdp_demo.md) | `all_gather` + `reduce_scatter` (shard params & grads) |
| **2. Process-group meshes** | ★★★☆☆ | [`5_sequence_parallel/1_megatron_sp.py`](5_sequence_parallel/1_megatron_sp.py) | [doc](5_sequence_parallel/1_megatron_sp.md) | the `all_gather`/`reduce_scatter` conjugate pair |
| | ★★★☆☆ | [`5_sequence_parallel/2_ulysses_sp.py`](5_sequence_parallel/2_ulysses_sp.py) | [doc](5_sequence_parallel/2_ulysses_sp.md) | `all_to_all` (swap seq ↔ head) |
| | ★★★☆☆ | [`1_data_parallel/4_hsdp_demo.py`](1_data_parallel/4_hsdp_demo.py) | [doc](1_data_parallel/4_hsdp_demo.md) | build subgroups with `new_group` (2D mesh) |
| | ★★★☆☆ | [`2_tensor_parallel/3_summa_2d.py`](2_tensor_parallel/3_summa_2d.py) | [doc](2_tensor_parallel/3_summa_2d.md) | row/column `broadcast` on a √N grid |
| **3. Pipeline state machines** | ★★★☆☆ | [`3_pipeline_parallel/1_gpipe.py`](3_pipeline_parallel/1_gpipe.py) | [doc](3_pipeline_parallel/1_gpipe.md) | raw `send`/`recv` + F-then-B schedule |
| | ★★★★☆ | [`3_pipeline_parallel/2_one_forward_backward.py`](3_pipeline_parallel/2_one_forward_backward.py) | [doc](3_pipeline_parallel/2_one_forward_backward.md) | the 1F1B warmup/steady/cooldown machine |
| | ★★★★☆ | [`3_pipeline_parallel/4_pipedream.py`](3_pipeline_parallel/4_pipedream.py) | [doc](3_pipeline_parallel/4_pipedream.md) | async schedule + weight stashing |
| **4. MoE & attention frontier** | ★★★☆☆ | [`7_multi_head_latent_attention/1_deepseek_mla.py`](7_multi_head_latent_attention/1_deepseek_mla.py) | [doc](7_multi_head_latent_attention/1_deepseek_mla.md) | low-rank KV compression |
| | ★★★★☆ | [`4_expert_parallel/1_naive_moe.py`](4_expert_parallel/1_naive_moe.py) | [doc](4_expert_parallel/1_naive_moe.md) | token routing + `all_to_all` dispatch/combine |
| | ★★★★☆ | [`4_expert_parallel/2_lightning_moe.py`](4_expert_parallel/2_lightning_moe.py) | [doc](4_expert_parallel/2_lightning_moe.md) | tile-level comm/compute overlap |
| | ★★★★☆ | [`6_context_parallel/1_ring_attention.py`](6_context_parallel/1_ring_attention.py) | [doc](6_context_parallel/1_ring_attention.md) | online softmax + K/V ring rotation |
| **5. Advanced schedules & composition** | ★★★★★ | [`3_pipeline_parallel/3_interleaved_1f1b.py`](3_pipeline_parallel/3_interleaved_1f1b.py) | [doc](3_pipeline_parallel/3_interleaved_1f1b.md) | virtual pipeline (V chunks/rank) |
| | ★★★★★ | [`3_pipeline_parallel/5_chimera.py`](3_pipeline_parallel/5_chimera.py) | [doc](3_pipeline_parallel/5_chimera.md) | two opposing pipelines |
| | ★★★★★ | [`3_pipeline_parallel/6_deepseek_dualpipe.py`](3_pipeline_parallel/6_deepseek_dualpipe.py) | [doc](3_pipeline_parallel/6_deepseek_dualpipe.md) | bidirectional pipeline + overlap |
| | ★★★★★ | [`4_expert_parallel/3_deepseek_moe.py`](4_expert_parallel/3_deepseek_moe.py) | [doc](4_expert_parallel/3_deepseek_moe.md) | fine-grained + shared experts, aux-loss-free balance |
| | ★★★★★ | [`4_expert_parallel/4_deepseek_deepep.py`](4_expert_parallel/4_deepseek_deepep.py) | [doc](4_expert_parallel/4_deepseek_deepep.md) | overlapped 2-tier `all_to_all` |
| | ★★★★★ | [`8_hybrid_parallel/1_three_d_parallel.py`](8_hybrid_parallel/1_three_d_parallel.py) | [doc](8_hybrid_parallel/1_three_d_parallel.md) | **capstone**: DP × TP × PP on one mesh |
| | ★★★★★ | [`8_hybrid_parallel/2_nd_mesh_parallel.py`](8_hybrid_parallel/2_nd_mesh_parallel.py) | [doc](8_hybrid_parallel/2_nd_mesh_parallel.md) | **grand capstone**: full 6D mesh `[TP,EP,CP,FS,DP,PP]` + per-axis flash |

**Rule of thumb**: finish all of Level 0 first (they each take ~10 lines and teach one
collective), then climb. Within a module always do the **baseline before the evolution**
and diff them -- that contrast is where the "aha" lives. The `deepseek_*` and Level-5
demos assume you've already internalized the baseline they build on.

Every demo also has a companion `*.md` doc next to it (algorithm, paper references, and a
visualization) -- read it before you start coding.

---

## The only API you need to know: `launch_teaching_cluster`

Every demo's bottom looks like this; you just write `run`:

```python
from env_setup import launch_teaching_cluster

def run(rank, world_size, device):
    ...  # your distributed core logic (compute on device; .to(COMM_DEVICE) before communicating)

if __name__ == "__main__":
    launch_teaching_cluster(world_size=4, func=run)
```

`launch_teaching_cluster` handles all the dirty work: auto-detect the backend
(NCCL/gloo), auto-find a free port, set `MASTER_ADDR/PORT`, `init_process_group`, bind a
device per rank (`cuda:i` / shared MPS / CPU), and cleanly `barrier` +
`destroy_process_group` when done.

`env_setup` also hands you small dashboard tools: `rank_print` / `rank0_print` (colored,
timestamped multi-process logs), `banner` / `bar` (ASCII banners and comparison bars),
`Timeline` + `render_gantt` (a Gantt-chart timeline base), and `build_mesh` / `render_mesh`
(compose an N-D device mesh and flash its per-axis process groups, used by module 8).

---

## Five geeky selling points

1. **One-command run, any hardware**: `mp.spawn` is built in; `python xxx_demo.py` spins up
   the processes directly, zero environment variables. The backend auto-adapts -- NCCL on a
   GPU box, gloo + MPS on a Mac, gloo + CPU otherwise.
2. **Artificial topology simulation (Traffic Shaper)**: `topology_sim.py` uses `time.sleep`
   to manufacture a "cross-machine IB NIC" bottleneck inside the Mac's fast unified memory,
   specifically to showcase the power of Tile Overlap.
3. **Pixel-level raw writing, no abstraction hell**: no `DistributedDataParallel`, no
   `nn.Sequential` black boxes -- all `dist.*` communication is written out in the main line
   of forward/backward, every line of communication visible.
4. **Flashy terminal Timeline dashboards**: pure Python + ANSI draw 1F1B's `[F1][F2][B1]...`
   conveyor timing and the ASCII performance comparison bar showing MegaScale's comm bubble
   getting filled before vs after.
5. **The long-context frontier (SP & CP)**: Sequence Parallel in two flavors -- Megatron's
   All-Gather/Reduce-Scatter conjugate pair and DeepSpeed-Ulysses' All-to-All head/sequence
   swap -- plus Context Parallel's Ring Attention with online softmax: the techniques behind
   training on million-token sequences.

## The full parallelism zoo

Beyond the four core axes, the repo now hand-writes the rest of the industry-standard
parallelism strategies, each as a "baseline -> evolution" twin you can diff:

- **Memory sharding ladder** (Module 1): DDP -> ZeRO-1 -> **ZeRO-3/FSDP** (param + grad
  All-Gather/Reduce-Scatter) -> **HSDP** (2D mesh: intra-node FSDP, inter-node DDP).
- **Tensor parallel** (Module 2): 1D column/row split -> **2D/2.5D SUMMA** on a sqrt(N) grid.
- **Pipeline schedules** (Module 3): GPipe -> 1F1B -> **Interleaved 1F1B** (Megatron virtual
  pipeline) -> **PipeDream** (async + weight stashing) -> **Chimera** (bidirectional) ->
  DeepSeek DualPipe.
- **Capstones** (Module 8): compose the axes on a single device mesh.
  [`1_three_d_parallel.py`](8_hybrid_parallel/1_three_d_parallel.py) wires DP x TP x PP (TP
  All-Reduce inside a layer, PP send/recv across stages, DP All-Reduce across replicas);
  [`2_nd_mesh_parallel.py`](8_hybrid_parallel/2_nd_mesh_parallel.py) builds the full **6D**
  mesh `[TP, EP, CP, FS, DP, PP]` and flashes which process group each axis communicates over
  -- a terminal take on [Visualizing 6D Mesh Parallelism](https://main-horse.github.io/posts/visualizing-6d/).

---

## DeepSeek-specific algorithms

Four flagship DeepSeek-V2/V3 techniques are included as `deepseek_*` demos, each layered on
the matching base module:

- **DualPipe** ([`3_pipeline_parallel/6_deepseek_dualpipe.py`](3_pipeline_parallel/6_deepseek_dualpipe.py)) -- bidirectional pipeline that runs
  two opposing micro-batch streams to fill each other's bubbles, on top of comm/compute overlap.
- **DeepSeekMoE** ([`4_expert_parallel/3_deepseek_moe.py`](4_expert_parallel/3_deepseek_moe.py)) -- many fine-grained routed experts plus
  always-on shared experts, with **auxiliary-loss-free** load balancing (a per-expert bias nudged
  from the global load via All-Reduce, instead of an auxiliary loss).
- **DeepEP** ([`4_expert_parallel/4_deepseek_deepep.py`](4_expert_parallel/4_deepseek_deepep.py)) -- overlap BOTH the dispatch and combine
  All-to-All across token chunks, modeling the intra-node (NVLink) vs inter-node (RDMA) tiers.
- **MLA** ([`7_multi_head_latent_attention/1_deepseek_mla.py`](7_multi_head_latent_attention/1_deepseek_mla.py)) -- Multi-head Latent Attention, which
  caches a small low-rank latent instead of full K/V, shrinking the KV cache dramatically.

---

## Notes for the "teacher/author"

- `env_setup/__init__.py` and `topology_sim.py` are **finished foundations**, usable as-is.
- All other `*.py` files leave a `# TODO(you)` battle zone at the core, with detailed
  step-by-step comments. Just replace the `raise NotImplementedError(...)` lines with your
  implementation.
- `Timeline` / `render_gantt` is only the most bare-bones dashboard base -- making it
  flashier (per selling-point #4) is an easter egg left for you.

> Note on layout: the root folders use numeric prefixes like `1_data_parallel` (not valid
> Python package names), so each demo adds one `sys.path` bootstrap line at the top to
> `import env_setup`. That is deliberate, to keep the "clone-and-run, zero-config" promise.

---

*Happy hacking.*
