"""Command-line entry point for micro-trainer.

This package itself carries no teaching logic -- the real content is the modules in
the repository root (1_data_parallel / 2_tensor_parallel / 3_pipeline_parallel /
4_expert_parallel / 5_sequence_parallel / 6_context_parallel /
7_multi_head_latent_attention / 8_hybrid_parallel). The `micro-trainer` command is
just a navigation map telling you where to start.
"""

from __future__ import annotations

DEMOS = [
    ("1_data_parallel/ddp_demo.py", "Naive DDP (hand-written gradient All-Reduce averaging)"),
    ("1_data_parallel/zero1_demo.py", "ZeRO-1 (optimizer state sharding + All-Gather)"),
    ("1_data_parallel/fsdp_demo.py", "ZeRO-3 / FSDP (param + grad sharding, All-Gather + Reduce-Scatter)"),
    ("1_data_parallel/hsdp_demo.py", "HSDP (2D mesh: intra-node FSDP + inter-node DDP)"),
    ("2_tensor_parallel/column_parallel.py", "Column-parallel linear (All-Gather concat)"),
    ("2_tensor_parallel/row_parallel.py", "Row-parallel linear (All-Reduce sum)"),
    ("2_tensor_parallel/summa_2d.py", "2D/2.5D SUMMA tensor parallel (grid broadcasts)"),
    ("3_pipeline_parallel/gpipe.py", "GPipe schedule (big Bubble pain point)"),
    ("3_pipeline_parallel/one_forward_backward.py", "1F1B schedule (live state machine)"),
    ("3_pipeline_parallel/interleaved_1f1b.py", "Interleaved 1F1B (Megatron virtual pipeline)"),
    ("3_pipeline_parallel/pipedream.py", "PipeDream (async pipeline + weight stashing)"),
    ("3_pipeline_parallel/chimera.py", "Chimera (bidirectional pipeline)"),
    ("3_pipeline_parallel/deepseek_dualpipe.py", "[DeepSeek] DualPipe (bidirectional pipeline overlap)"),
    ("4_expert_parallel/naive_moe.py", "Serial MoE / Expert Parallel (All-to-All dumb waiting)"),
    ("4_expert_parallel/lightning_moe.py", "Lightning MoE (Tile-level comm overlap)"),
    ("4_expert_parallel/deepseek_moe.py", "[DeepSeek] DeepSeekMoE (fine-grained + shared, aux-loss-free balancing)"),
    ("4_expert_parallel/deepseek_deepep.py", "[DeepSeek] DeepEP (overlapped dispatch + combine All-to-All)"),
    ("5_sequence_parallel/megatron_sp.py", "Megatron SP (All-Gather / Reduce-Scatter)"),
    ("5_sequence_parallel/ulysses_sp.py", "DeepSpeed-Ulysses SP (All-to-All head/seq swap)"),
    ("6_context_parallel/ring_attention.py", "Context Parallel (Ring Attention + online softmax)"),
    ("7_multi_head_latent_attention/deepseek_mla.py", "[DeepSeek] MLA (low-rank KV compression)"),
    ("8_hybrid_parallel/three_d_parallel.py", "3D Parallelism capstone (DP x TP x PP composed)"),
]


def main() -> None:
    print("micro-trainer -- hand-write large-model distributed training on a Mac\n")
    print("Pick a demo and run it directly (no environment variables needed):\n")
    for path, desc in DEMOS:
        print(f"  python {path:<52} # {desc}")
    print("\nSee README.md for details. Happy hacking!")


if __name__ == "__main__":
    main()
