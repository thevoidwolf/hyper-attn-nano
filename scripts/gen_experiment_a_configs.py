#!/usr/bin/env python3
"""
Generate all main-grid YAML configs for Experiment A.
Run once from the project root:
    python scripts/gen_experiment_a_configs.py
"""

import os
import textwrap

OUT_DIR = "configs/experiment_a"
os.makedirs(OUT_DIR, exist_ok=True)

SCALES = {
    "S1": {"d_model": 64,  "d_ff": 256,  "n_heads": 2},
    "S2": {"d_model": 128, "d_ff": 512,  "n_heads": 4},
    "S3": {"d_model": 256, "d_ff": 1024, "n_heads": 8},
    "S4": {"d_model": 384, "d_ff": 1536, "n_heads": 12},
    "S5": {"d_model": 512, "d_ff": 2048, "n_heads": 16},
}

SEEDS = [42, 1337, 2718]


def euclid_config(scale_tag: str, d: int, d_ff: int, n_heads: int, seed: int) -> str:
    s = scale_tag[1]
    run_id = f"a_s{s}_euclid_seed{seed}"
    return textwrap.dedent(f"""\
        # Experiment A main grid — Euclidean baseline, scale {scale_tag}, seed {seed}

        experiment:   experiment_a
        spec_version: "A.1.1"
        scale_tag:    {scale_tag}

        model:
          type:        euclid
          d_model:     {d}
          n_layers:    4
          n_heads:     {n_heads}
          d_ff:        {d_ff}
          max_seq_len: 256
          vocab_size:  50257

        training:
          dataset:              wikitext2
          data_dir:             data/wikitext2
          batch_size:           32
          max_steps:            9000
          eval_interval:        250
          checkpoint_interval:  500
          log_interval:         50
          eval_batches:         50
          full_split_eval:      true
          lr:                   3.0e-4
          weight_decay:         0.1
          warmup_steps:         500
          min_lr_ratio:         0.1
          seed:                 {seed}

        run_id:         "{run_id}"
        log_dir:        "logs/experiment_a/{run_id}"
        checkpoint_dir: "checkpoints/experiment_a/{run_id}"
        notes:          "Experiment A main grid — Euclidean, scale {scale_tag}, seed {seed}."
        """)


def hyper_config(scale_tag: str, d: int, d_ff: int, n_heads: int, seed: int) -> str:
    s = scale_tag[1]
    run_id = f"a_s{s}_hyper_seed{seed}"
    note = ""
    if scale_tag == "S2" and seed == 42:
        note = " Also serves as Preflight 1b (spec §17.4)."
    if scale_tag == "S1":
        note += " n_heads=2 (head_dim=32) — intentional change from old 2M config (spec §2)."
    return textwrap.dedent(f"""\
        # Experiment A main grid — hyperbolic (hyper-fixed), scale {scale_tag}, seed {seed}
        #{note}

        experiment:   experiment_a
        spec_version: "A.1.1"
        scale_tag:    {scale_tag}

        model:
          type:             hyper-fixed
          d_model:          {d}
          n_layers:         4
          n_heads:          {n_heads}
          d_ff:             {d_ff}
          max_seq_len:      256
          vocab_size:       50257
          manifold_float64: true
          curvature:        -10.0
          curvature_schedule:
            type:         linear_warmup
            k_start:      -1.0
            k_end:        -10.0
            warmup_steps: 500

        training:
          dataset:              wikitext2
          data_dir:             data/wikitext2
          batch_size:           32
          max_steps:            9000
          eval_interval:        250
          checkpoint_interval:  500
          log_interval:         50
          eval_batches:         50
          full_split_eval:      true
          extra_cosine_decay:   true
          lr:                   6.0e-4
          weight_decay:         0.1
          warmup_steps:         500
          min_lr_ratio:         0.1
          seed:                 {seed}

        run_id:         "{run_id}"
        log_dir:        "logs/experiment_a/{run_id}"
        checkpoint_dir: "checkpoints/experiment_a/{run_id}"
        notes:          "Experiment A main grid — hyper-fixed K=-10, scale {scale_tag}, seed {seed}.{note}"
        """)


generated = []
for scale_tag, params in SCALES.items():
    d, d_ff, n_heads = params["d_model"], params["d_ff"], params["n_heads"]
    s = scale_tag[1]
    for seed in SEEDS:
        # Euclidean
        path_e = os.path.join(OUT_DIR, f"a_s{s}_euclid_seed{seed}.yaml")
        with open(path_e, "w") as f:
            f.write(euclid_config(scale_tag, d, d_ff, n_heads, seed))
        generated.append(path_e)

        # Hyperbolic
        path_h = os.path.join(OUT_DIR, f"a_s{s}_hyper_seed{seed}.yaml")
        with open(path_h, "w") as f:
            f.write(hyper_config(scale_tag, d, d_ff, n_heads, seed))
        generated.append(path_h)

print(f"Generated {len(generated)} configs in {OUT_DIR}/")
for p in sorted(generated):
    print(f"  {p}")
