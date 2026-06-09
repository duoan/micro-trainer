# ZeRO-2: Gradient Sharding (a single ShardedDDP wrapper)

> ZeRO-2 goes one step past ZeRO-1's sharded *optimizer*: a single `MicroShardedDistributedDataParallel` wrapper internally shards **both** the optimizer state **and** the gradients — backward hooks reduce each grad to its owner, and the wrapper owns the sharded optimizer too.

## TL;DR

- ZeRO-1 (demo 4) is just an **optimizer**: it shards optimizer state but `all_reduce`s grads, so every rank still holds the **full gradient**.
- ZeRO-2 folds everything into one **wrapper**: a backward hook fires **`reduce(grad, dst=owner)`** per parameter, summing each grad onto exactly one rank; non-owners drop theirs.
- The wrapper also builds the base optimizer over only the owned params, so **optimizer state and gradients are both sharded ≈ $1/N$** — handled by one object.
- Parameters are still replicated — sharding those too is ZeRO-3 / FSDP (next demo).

## Why one wrapper?

Gradients become ready *during* the backward pass, so the natural place to reduce them is a backward hook — exactly like the DDP overlap demo (Level 2), where comm overlaps with the rest of backward. Optimizer-state sharding is a constructor-time decision (which params do I own?), so the same object can own the sharded optimizer and apply it after the grads land. Keeping it all in one `MicroShardedDistributedDataParallel` means a single class tells the whole ZeRO-2 story.

> FairScale splits this into [`OSS`](https://github.com/facebookresearch/fairscale/blob/main/fairscale/optim/oss.py) (sharded optimizer) + [`ShardedDataParallel`](https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/data_parallel/sharded_ddp.py) (gradient-reducing wrapper). We fold both into one class for teaching clarity; the gradient-reduction logic is the same `dist.reduce(grad, dst=owner, async_op=True)`.

## Algorithm

```
ddp = MicroShardedDistributedDataParallel(model, world_size, rank, Adam, lr=...)

ddp.zero_grad()
loss = loss_fn(ddp(x), y)
loss.backward()    # (a) hooks reduce each grad toward its owner, async
ddp.step()         #     finish_reduce (b) -> sharded opt.step() -> broadcast params
```

1. **Partition + sharded optimizer** (`__init__`): `owner[i] = i % world_size`; broadcast initial weights; build the base optimizer over only the owned params (optimizer-state sharding); register a backward hook on every parameter.
2. **Backward hook** (`_reduce_hook`, battle zone a): when grad $i$ is ready, scale it by $1/N$ and **`dist.reduce(grad, dst=owner[i], op=SUM, async_op=True)`** — the sum lands on the owner only; record the work handle.
3. **Drain + shard** (`finish_reduce`, battle zone b): wait on every handle, then **free `p.grad` on every rank that isn't the owner** — the gradient-memory saving made real.
4. **Sharded step** (`step`, provided): `self.opt.step()` updates owned params from their reduced grad, then broadcast each param from its owner so every rank holds the full weights.

## Communication pattern

```mermaid
sequenceDiagram
    participant BW as backward()
    participant H as grad hooks (ShardedDDP)
    participant C as comm (async)

    BW->>H: grad of param i ready
    H->>C: reduce(grad_i, dst=owner[i], async)  ── sum lands on the owner ONLY
    BW->>H: grad of param j ready
    H->>C: reduce(grad_j, dst=owner[j], async)
    Note over BW,C: reduces overlap with the rest of backward (like Level 2)
    Note over H,C: step(): wait all; non-owners free their grads
    Note over H,C: opt.step() on owned shard, then broadcast params back
```

## What you'll see

Each rank prints which parameter indices it owns, e.g. `params I own = [1] (of 4 tensors) -> only these grads survive backward`. Before implementation the run stops at:

```
NotImplementedError: TODO(a): reduce each grad toward its owner in _reduce_hook
```

then at battle zone (b) in `finish_reduce`. When both are correct, rank 0 prints a cross-rank parameter drift of `0.00e+00` — every rank ends the step with bit-identical weights, having reduced each grad straight to its owner during backward and never stored a gradient it didn't own.

## Your battle zone

Two methods of `MicroShardedDistributedDataParallel` in `1_data_parallel/5_zero2_demo.py` (the partition, the internal sharded optimizer, `step`, and `_sync_params` are provided):

**1. `_reduce_hook(param)`** — fires when `param.grad` is ready. Scale the grad by `1 / world_size`, launch `dist.reduce(param.grad, dst=self._owner_of[param], op=dist.ReduceOp.SUM, async_op=True)`, and stash the returned handle (plus the param + its owner) for `finish_reduce`.

**2. `finish_reduce()`** — wait on every handle, then set `p.grad = None` for every param this rank does not own (the memory win), and clear the handle list.

## Run it

```bash
python 1_data_parallel/5_zero2_demo.py
```

## Papers & further reading

- Rajbhandari et al., 2020, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (SC20) — stage $P_{os+g}$ (ZeRO-2).
- FairScale [`ShardedDataParallel`](https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/data_parallel/sharded_ddp.py) + `OSS` — the production version of this idea (split across two objects).
- PyTorch docs: `Tensor.register_post_accumulate_grad_hook`, `torch.distributed.reduce` (`dst=`, `async_op=True`).
