"""Command-line entry point for micro-trainer.

This package itself carries no teaching logic -- the real content is the modules in
the repository root (1_data_parallel / 2_tensor_parallel / 3_pipeline_parallel /
4_expert_parallel / 5_sequence_parallel / 6_context_parallel /
7_multi_head_latent_attention / 8_hybrid_parallel). The `micro-trainer` command is
just a navigation map telling you where to start.
"""

from __future__ import annotations

DEMOS = [
    ("1_data_parallel/1_ddp_demo.py", "Naive DDP as a MicroDDP module (one All-Reduce per param)"),
    ("1_data_parallel/2_ddp_bucketing.py", "DDP gradient bucketing (one fused All-Reduce)"),
    ("1_data_parallel/3_ddp_overlap.py", "DDP comm/compute overlap (async All-Reduce from backward hooks)"),
    ("1_data_parallel/4_zero1_demo.py", "ZeRO-1 (optimizer state sharding + All-Gather)"),
    (
        "1_data_parallel/5_fsdp_demo.py",
        "ZeRO-3 / FSDP (param + grad sharding, All-Gather + Reduce-Scatter)",
    ),
    ("1_data_parallel/6_hsdp_demo.py", "HSDP (2D mesh: intra-node FSDP + inter-node DDP)"),
    ("2_tensor_parallel/1_column_parallel.py", "Column-parallel linear (All-Gather concat)"),
    ("2_tensor_parallel/2_row_parallel.py", "Row-parallel linear (All-Reduce sum)"),
    ("2_tensor_parallel/3_summa_2d.py", "2D/2.5D SUMMA tensor parallel (grid broadcasts)"),
    ("3_pipeline_parallel/1_gpipe.py", "GPipe schedule (big Bubble pain point)"),
    ("3_pipeline_parallel/2_one_forward_backward.py", "1F1B schedule (live state machine)"),
    ("3_pipeline_parallel/3_interleaved_1f1b.py", "Interleaved 1F1B (Megatron virtual pipeline)"),
    ("3_pipeline_parallel/4_pipedream.py", "PipeDream (async pipeline + weight stashing)"),
    ("3_pipeline_parallel/5_chimera.py", "Chimera (bidirectional pipeline)"),
    (
        "3_pipeline_parallel/6_deepseek_dualpipe.py",
        "[DeepSeek] DualPipe (bidirectional pipeline overlap)",
    ),
    ("4_expert_parallel/1_naive_moe.py", "Serial MoE / Expert Parallel (All-to-All dumb waiting)"),
    ("4_expert_parallel/2_lightning_moe.py", "Lightning MoE (Tile-level comm overlap)"),
    (
        "4_expert_parallel/3_deepseek_moe.py",
        "[DeepSeek] DeepSeekMoE (fine-grained + shared, aux-loss-free balancing)",
    ),
    (
        "4_expert_parallel/4_deepseek_deepep.py",
        "[DeepSeek] DeepEP (overlapped dispatch + combine All-to-All)",
    ),
    ("5_sequence_parallel/1_megatron_sp.py", "Megatron SP (All-Gather / Reduce-Scatter)"),
    ("5_sequence_parallel/2_ulysses_sp.py", "DeepSpeed-Ulysses SP (All-to-All head/seq swap)"),
    (
        "6_context_parallel/1_ring_attention.py",
        "Context Parallel (Ring Attention + online softmax)",
    ),
    ("7_multi_head_latent_attention/1_deepseek_mla.py", "[DeepSeek] MLA (low-rank KV compression)"),
    ("8_hybrid_parallel/1_three_d_parallel.py", "3D Parallelism capstone (DP x TP x PP composed)"),
    (
        "8_hybrid_parallel/2_nd_mesh_parallel.py",
        "N-D (6D) mesh capstone (TP/EP/CP/FS/DP/PP, flash per axis)",
    ),
]


def main() -> None:
    print("micro-trainer -- hand-write large-model distributed training on a Mac\n")
    print("Pick a demo and run it directly (no environment variables needed):\n")
    for path, desc in DEMOS:
        print(f"  python {path:<52} # {desc}")
    print("\nSee README.md for details. Happy hacking!")


if __name__ == "__main__":
    main()
