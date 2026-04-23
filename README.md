# HyperAttn-Nano

A minimal proof-of-concept decoder-only language model with **per-head learnable curvature in hyperbolic attention**. Built to test whether giving each attention head its own curvature parameter improves on fixed-curvature hyperbolic attention (the approach used in [HELM](https://arxiv.org/abs/2505.24722)).

**Short answer: hyperbolic attention does not outperform Euclidean at this scale, and the gap does not close on hierarchically-structured data.** Three follow-up experiments tested optimal curvature (K=-10 is the float32 ceiling), float64 precision effects (overfits on prose, neutral on code), and hierarchical data (CodeParrot code widens the euclid–hyperbolic gap). Per-head learnable curvature produces a novel layer-stratification pattern on prose that disappears on code. See [FINDINGS.md](FINDINGS.md) for the full write-up.

---

## Background

Hyperbolic space can represent hierarchical structure exponentially more efficiently than flat (Euclidean) space — a 2D hyperbolic embedding can match a 200D Euclidean embedding on tree-structured data (Nickel & Kiela, 2017). Language has hierarchical structure, so the theoretical case for hyperbolic attention in LLMs is real.

**The gap in prior work:** HELM (NeurIPS 2025) introduced the first fully hyperbolic decoder-only LLMs at scale. Its attention mechanism uses a *fixed* curvature. The curvature diversity in HELM lives in the FFN (MoE) layer, not in the attention heads.

**Our hypothesis:** Different attention heads specialise in different depths of the semantic hierarchy. A head tracking local syntax should want near-flat geometry; a head tracking long-range semantic relationships should want stronger curvature. Giving each head a *learnable* curvature K lets the model discover this automatically — and we can measure what it learns.

---

## The three model variants

| Variant | Description |
|---|---|
| `euclid` | Standard GPT-style decoder, dot-product attention (Euclidean baseline) |
| `hyper-fixed` | Hyperbolic decoder, all heads fixed at K=-1.0 (HELM-D style) |
| `hyper-perhead` | Hyperbolic decoder, per-head learnable K (novel contribution) |

All variants share the same architecture (NANO_CONFIG: d_model=128, 4 layers, 4 heads, d_ff=512), parameter count, dataset (WikiText-2), and random seed.

---

## Results summary

### Phase 1 — Per-head vs fixed curvature (V1/V2/V3, WikiText-2)

Three training versions, each correcting a fairness issue identified in the previous one.

| Version | What changed | euclid PPL | hyper-fixed PPL | hyper-perhead PPL |
|---|---|---|---|---|
| V1 (5k steps, LR=3e-4 all) | Baseline | 277 | 451 | 450 |
| V2 (5k steps, hyper LR=6e-4) | Equalised gradient scale | 277 | 323 | 321 |
| V3 (9k steps, hyper LR=6e-4) | Equalised total gradient signal | 277 | 311 | 312 |

After gradient-equivalent training, the hyperbolic variants trail euclid by ~34 PPL. The V2 advantage of per-head over fixed (2.4 PPL) reversed in V3 (1.2 PPL).

### Phase 2 — Experiment B: forced curvature sweep (WikiText-2, float32)

V3 per-head curvatures converged near K≈-1.1. Experiment B tested whether forcing stronger fixed curvatures improves PPL. Stability improvements to `manifolds.py` (explicit angle clamp) also raised the K=-1 baseline by ~14 PPL over V3, confirming V3 had residual numerical instability.

| Run | K | PPL |
|-----|---|-----|
| exp_b_k1 | -1.0 | 296.1 |
| exp_b_k2 | -2.0 | 296.3 |
| exp_b_k5 | -5.0 | 288.0 |
| exp_b_k10 | **-10.0** | **283.7** |
| exp_b_k50 | -50.0 | 296.6 |

euclid baseline: 277.1 PPL. K=-10 closes the gap to 6.6 PPL. K=-50 degrades due to float32 angle clamping.

### Phase 3 — Experiment B2: float64 high-curvature probe (WikiText-2)

Surgical float64 promotion in manifold ops only, to extend beyond the float32 ceiling. Replication check halted the sweep.

| Run | K | Precision | PPL |
|-----|---|-----------|-----|
| B k10 (baseline) | -10.0 | float32 | 283.7 |
| B2 k10 | -10.0 | float64 | **301.4** |

Float64 at K=-10 is 17.7 PPL *worse* than float32. The float32 angle clamp acts as an implicit regularizer on WikiText-2; removing it causes overfitting.

### Phase 4 — Experiment D: hierarchical data probe (CodeParrot)

All four variants trained on Python source code (~2.4M tokens, matched to WikiText-2 size), tokenised with the same GPT-2 vocabulary.

| Run | Variant | PPL |
|-----|---------|-----|
| exp_d_euclid | Euclid | **26.2** |
| exp_d_k10_f32 | Hyper fixed K=-10 float32 | 39.9 |
| exp_d_k10_f64 | Hyper fixed K=-10 float64 | 39.9 |
| exp_d_perhead_k10 | Hyper perhead init K=-10 | 34.9 |

Cross-dataset gap comparison (Hyper fixed K=-10 float32 vs euclid):

| Dataset | Euclid PPL | Hyper PPL | Gap |
|---------|-----------|-----------|-----|
| WikiText-2 | 277.1 | 283.7 | +6.6 |
| CodeParrot | 26.2 | 39.9 | **+13.7** |

The euclid–hyperbolic gap is larger on code, not smaller. See [FINDINGS.md](FINDINGS.md) for the full analysis.

---

## Key findings

### Finding 1: Curvature layer stratification on WikiText-2

Across all three V1/V2/V3 runs, `hyper-perhead` developed a stable layer-wise pattern independently each time:

- **Layer 2 (penultimate) was always most curved** — K reaching -1.24 by V3
- **Layer 1 was always flattest** — K staying close to -1.06
- The spread across all heads grew with each version: 0.117 (V1) → 0.145 (V2) → 0.175 (V3)
- The same head (`layer_2_head_2`) was the most curved head in every version

This pattern has not been documented in prior literature. HELM uses fixed curvature in attention, so per-head specialisation has never been measurable in a prior published system.

### Finding 2: K=-10 is the float32 optimum on WikiText-2

Increasing fixed curvature from K=-1 to K=-10 closes 12.4 PPL of the euclid–hyperbolic gap. Beyond K=-10, float32 angle clamping prevents the geometry from being fully realised. K=-50's effective curvature under float32 is estimated at K=-12 to K=-20.

### Finding 3: Float64 precision overfits on shallow-hierarchy text

Float64 manifold precision at K=-10 worsens WikiText-2 PPL by 17.7 points over float32. The float32 angle clamp functions as an implicit regularizer: removing it allows the geometry to memorise training patterns in prose, increasing validation perplexity. This is a counterintuitive precision–regularization trade-off.

### Finding 4: The euclid–hyperbolic gap is larger on code

The hypothesis that code's deep parse-tree structure would help hyperbolic attention was wrong. The gap grew from 6.6 PPL (WikiText-2) to 13.7 PPL (CodeParrot) at K=-10. Code's rigid syntactic structure appears to benefit Euclidean attention more than hyperbolic.

### Finding 5: Float64 does not overfit on code; per-head wins by 5 PPL

On CodeParrot, float32 and float64 give identical PPL — the B2 overfitting was WikiText-2-specific. Per-head curvatures initialised at K=-10 converge to K≈-6 and gain 5.0 PPL over fixed K=-10 (34.9 vs 39.9), showing that learnable curvature provides a genuine search advantage when initialised in a useful curvature regime.

### Finding 6: Stratification disappears on code

On CodeParrot, per-head curvatures initialised at K=-10 converge to a uniform distribution (all layers K≈-6, no layer-level structure). The WikiText-2 stratification pattern is not a general property of hyperbolic attention — it is specific to the K=-1 initialisation regime on prose data.

---

## Repo structure

```
hyper-attn-nano/
├── src/
│   ├── manifolds.py          Lorentz manifold ops (exp_map, log_map, inner product)
│   ├── attention.py          LorentzPerHeadAttention + EuclideanAttention
│   ├── blocks.py             LorentzRMSNorm, LorentzFFN, decoder block
│   └── model.py              HyperAttnNano + GPTNano baseline
├── scripts/
│   ├── data_prep.py          Download + tokenise WikiText-2
│   ├── data_prep_code.py     Download + tokenise CodeParrot (Experiment D)
│   ├── train.py              Training loop (all variants)
│   ├── sweep_curvature.py    Automated sweep runner for Experiment B
│   ├── sweep_b2.py           Float64 sweep runner for Experiment B2
│   ├── run_exp_d.py          Automated runner for Experiment D variants
│   ├── compare.py            PPL comparison plots (V1/V2/V3)
│   ├── plot_curvature_sweep.py  Experiment B sweep plots
│   ├── plot_b2.py            Experiment B2 result plots
│   ├── plot_exp_d.py         Experiment D result plots
│   └── viz_curvature.py      Per-head curvature evolution + heatmap plots
├── configs/
│   ├── nano_euclid.yaml / nano_*_v2.yaml / nano_*_v3.yaml   V1/V2/V3 configs
│   ├── nano_hyper_fixed_k{1,2,5,10,50}.yaml                 Experiment B configs
│   ├── nano_b2_k{10,20,50,100}.yaml                         Experiment B2 configs
│   └── nano_{euclid,hyper_fixed_k10,hyper_fixed_k10_f64,hyper_perhead_k10}_code.yaml  Experiment D configs
├── data/
│   ├── wikitext2/            WikiText-2 tokenised (V1–V3, Experiment B/B2)
│   └── codeparrot/           CodeParrot tokenised (Experiment D)
├── results/
│   ├── logs/{v1,v2,v3}/      V1/V2/V3 training logs
│   ├── logs/exp_b/           Experiment B logs
│   ├── logs/exp_b2/          Experiment B2 logs
│   ├── logs/exp_d/           Experiment D logs
│   ├── exp_b_summary.md      Experiment B results table
│   ├── exp_b2_summary.md     Experiment B2 results table
│   ├── exp_d_summary.md      Experiment D results table
│   └── plots/                Generated plots
├── tests/                    Unit tests for all src/ modules
├── setup/smoke_test.py       Quick environment check
├── requirements.txt
└── FINDINGS.md               Full experimental write-up
```

---

## Setup

**Requirements:** Python 3.11, CUDA-capable GPU (tested on RTX 3060 12GB).

```bash
# 1. Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate   # Windows WSL2: same command

# 2. Install PyTorch (adjust for your CUDA version)
pip install torch==2.7.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Activate project environment (sets PYTHONPATH, pins GPU)
source activate.sh

# 5. Verify setup
python setup/smoke_test.py
```

---

## Reproducing the experiments

```bash
source activate.sh

# Prepare WikiText-2 (downloads ~2 min)
python scripts/data_prep.py --dataset wikitext2 --out data/wikitext2

# Prepare CodeParrot (downloads ~5 min, streaming)
python scripts/data_prep_code.py --out data/codeparrot

# V3: three-way comparison at K=-1 (WikiText-2)
python scripts/train.py --config configs/nano_euclid.yaml
python scripts/train.py --config configs/nano_hyper_fixed_v3.yaml
python scripts/train.py --config configs/nano_hyper_perhead_v3.yaml

# Experiment B: curvature sweep K=-1 through K=-50 (WikiText-2, float32)
python scripts/sweep_curvature.py           # runs k1 → k50, stops on divergence

# Experiment B2: float64 high-curvature probe (WikiText-2)
python scripts/sweep_b2.py                  # runs k10 replication check, then k20/k50/k100

# Experiment D: hierarchical data probe (CodeParrot)
python scripts/run_exp_d.py                 # runs all four D variants in sequence

# Individual D variant:
python scripts/train.py --config configs/nano_euclid_code.yaml
python scripts/train.py --config configs/nano_hyper_fixed_k10_code.yaml
python scripts/train.py --config configs/nano_hyper_fixed_k10_f64_code.yaml
python scripts/train.py --config configs/nano_hyper_perhead_k10_code.yaml

# Generate plots from existing logs
python scripts/compare.py
python scripts/viz_curvature.py --version v3
python scripts/plot_curvature_sweep.py
python scripts/plot_exp_d.py
```

Training time on RTX 3060: ~45 min for 5000 steps, ~80 min for 9000 steps.

---

## Running the tests

```bash
source activate.sh
pytest tests/
```

---

## References

- Nickel & Kiela (2017) — [Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039)
- He et al. (ICLR 2025) — [HyperCore](https://arxiv.org/abs/2504.08912) — Encoder-only hyperbolic transformers
- He et al. (NeurIPS 2025) — [HELM](https://arxiv.org/abs/2505.24722) — First hyperbolic decoder-only LLMs. Closest prior work; uses fixed curvature in attention.
- [Geoopt](https://github.com/geoopt/geoopt) — Riemannian optimisation for PyTorch
