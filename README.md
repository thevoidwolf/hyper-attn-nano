# HyperAttn-Nano

> **Status: archived exploration.** This repo is a completed investigation, not active research. Notes and findings are preserved as a starting point for anyone exploring hyperbolic attention in language models. See the [closing note](#closing-note) for context.

A minimal proof-of-concept decoder-only language model with **per-head learnable curvature in hyperbolic attention**. Built to test whether giving each attention head its own curvature parameter improves on fixed-curvature hyperbolic attention (the approach used in [HELM](https://arxiv.org/abs/2505.24722)).

**Short answer: hyperbolic attention does not outperform Euclidean at this scale, and the gap does not close on hierarchically-structured data.** Three follow-up experiments tested optimal curvature (K=-10 is the float32 ceiling), float64 precision effects (overfits on prose, neutral on code), and hierarchical data (CodeParrot code widens the euclid–hyperbolic gap). Per-head learnable curvature produces a novel layer-stratification pattern on prose that disappears on code. See [FINDINGS.md](FINDINGS.md) for the full write-up.

---

## Background

Hyperbolic space can represent hierarchical structure exponentially more efficiently than flat (Euclidean) space - a 2D hyperbolic embedding can match a 200D Euclidean embedding on tree-structured data (Nickel & Kiela, 2017). Language has hierarchical structure, so the theoretical case for hyperbolic attention in LLMs is real.

**Where this sits relative to HELM (NeurIPS 2025):** HELM demonstrated that fully hyperbolic decoder-only LLMs are viable at billion-parameter scale. HELM's attention uses a single *fixed* curvature; its curvature diversity lives in the FFN (Mixture-of-Curvature Experts), not in the attention heads. This experiment tests an orthogonal axis: per-head *learnable* curvature in the attention mechanism itself.

**Specific hypothesis:** Different attention heads specialise in different depths of the semantic hierarchy. A head tracking local syntax should want near-flat geometry; a head tracking long-range semantic relationships should want stronger curvature. Giving each head a learnable curvature K lets the model discover this automatically - and makes it possible to measure what it learned.

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

### Phase 1 - Per-head vs fixed curvature (V1/V2/V3, WikiText-2)

Three training versions, each correcting a fairness issue identified in the previous one.

| Version | What changed | euclid PPL | hyper-fixed PPL | hyper-perhead PPL |
|---|---|---|---|---|
| V1 (5k steps, LR=3e-4 all) | Baseline | 277 | 451 | 450 |
| V2 (5k steps, hyper LR=6e-4) | Equalised gradient scale | 277 | 323 | 321 |
| V3 (9k steps, hyper LR=6e-4) | Equalised total gradient signal | 277 | 311 | 312 |

After gradient-equivalent training, the hyperbolic variants trail euclid by ~34 PPL. The V2 advantage of per-head over fixed (2.4 PPL) reversed in V3 (1.2 PPL).

### Phase 2 - Experiment B: forced curvature sweep (WikiText-2, float32)

V3 per-head curvatures converged near K≈-1.1. Experiment B tested whether forcing stronger fixed curvatures improves PPL. Stability improvements to `manifolds.py` (explicit angle clamp) also raised the K=-1 baseline by ~14 PPL over V3, confirming V3 had residual numerical instability.

| Run | K | PPL |
|-----|---|-----|
| exp_b_k1 | -1.0 | 296.1 |
| exp_b_k2 | -2.0 | 296.3 |
| exp_b_k5 | -5.0 | 288.0 |
| exp_b_k10 | **-10.0** | **283.7** |
| exp_b_k50 | -50.0 | 296.6 |

euclid baseline: 277.1 PPL. K=-10 closes the gap to 6.6 PPL. K=-50 degrades due to float32 angle clamping.

### Phase 3 - Experiment B2: float64 high-curvature probe (WikiText-2)

Surgical float64 promotion in manifold ops only, to extend beyond the float32 ceiling. Replication check halted the sweep.

| Run | K | Precision | PPL |
|-----|---|-----------|-----|
| B k10 (baseline) | -10.0 | float32 | 283.7 |
| B2 k10 | -10.0 | float64 | **301.4** |

Float64 at K=-10 is 17.7 PPL *worse* than float32. The float32 angle clamp acts as an implicit regularizer on WikiText-2; removing it causes overfitting.

### Phase 4 - Experiment D: hierarchical data probe (CodeParrot)

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

### What failed (the original hypothesis)

- **Hyperbolic attention did not outperform Euclidean at nano scale on either dataset tested.** After gradient-equivalent training (V3), the gap was 34.7 PPL on WikiText-2 at K=-1. Stronger fixed curvature (K=-10) narrowed the gap to 6.6 PPL on WikiText-2 but did not close it.
- **The euclid–hyperbolic gap widened on hierarchically-structured code, not narrowed.** The hypothesis that code's deep parse-tree structure would favour hyperbolic attention was wrong: the gap grew from 6.6 PPL (WikiText-2) to 13.7 PPL (CodeParrot) at K=-10. Code's rigid syntactic patterns appear to suit Euclidean attention's inductive bias, not hyperbolic.
- **Per-head vs fixed curvature at K=-1 was within noise on WikiText-2.** The V2 advantage of per-head (2.4 PPL) reversed to a 1.2 PPL deficit in V3. Neither direction is a reliable signal at this scale.

### Novel empirical observations

- **Curvature layer stratification on prose.** Across three independent V1/V2/V3 training runs with different hyperparameters, `hyper-perhead` consistently placed the most curved heads in the penultimate layer (Layer 2) and the flattest heads in Layer 1. The same specific head (`layer_2_head_2`) was the most curved head in every version. Spread grew with each version: 0.117 (V1) → 0.145 (V2) → 0.175 (V3). HELM uses fixed curvature in attention, so per-head specialisation has not been measurable in a prior published system.
- **Float64 manifold precision overfits on prose.** At K=-10, float64 manifold ops worsen WikiText-2 validation PPL by 17.7 points over float32. The float32 angle clamp (`MAX_ANGLE=10.0`) functions as an implicit regulariser on shallow-hierarchy text: removing it allows the geometry to memorise training patterns. This is a counterintuitive precision–regularisation trade-off that may have relevance to other curved-space learning systems.
- **Per-head wins on code when initialised at K=-10.** On CodeParrot, per-head curvatures initialised at K=-10 relax to K≈-6 and reach 34.9 PPL - 5.0 PPL better than fixed K=-10 (39.9 PPL). This is the first run in the programme where per-head curvature clearly and consistently outperforms a fixed-curvature baseline.
- **Stratification is data-distribution-dependent.** The WikiText-2 layer pattern does not persist on code. On CodeParrot with K=-10 initialisation, all layers converge to a uniform mean of K≈-6 with no layer-level structure. The stratification appears tied to the prose distribution and/or the K=-1 initialisation regime - a clean disambiguation experiment remains untested.

### Other findings

- **K=-10 is the float32 optimum on WikiText-2.** Monotone improvement from K=-1 (296 PPL) to K=-10 (283.7 PPL), then reversal at K=-50 (296.6 PPL) caused by angle clamping. The model cannot actually access K=-50 geometry in float32; effective curvature is estimated K=-12 to K=-20.
- **Float32 and float64 give identical PPL on code at K=-10.** The B2 overfitting is WikiText-2-specific.

---

## Pre-registered Experiment A (scale sweep)

A separate pre-registered experiment was set up to test the original order-of-magnitude efficiency claim directly: whether a hyperbolic model at scale S_n could match the best-val PPL of a Euclidean model at scale S_{n+1} or S_{n+2}, yielding ~2–4× parameter efficiency on WikiText-2. Grid: S1–S5 (d_model 64 → 512) × 3 seeds × 2 variants = 30 runs, with full-split ordered evaluation and bootstrap confidence intervals.

**Status: main grid not executed.** Preflights 1, 1b, 2, and 2b completed; Amendment A.1.2 was committed based on preflight findings; the full main grid was not run.

**Preflight findings worth preserving:**

- **Scale-dependent numerical stability wall.** At S5 (d_model=512), K=-10 produced non-finite loss at ~32% through training and K=-15 at ~25% through training. At S1 (d_model=64), the same K values completed training and landed within a 4.2 PPL band. Grad norms at crash time were well within the 1.0 clip (~0.9–1.1), indicating the NaN originates in the forward-pass manifold operations, not in gradient explosion. **The optimal K decreases with model scale** - the literature's default K=-1 is not merely suboptimal but the opposite direction of the scale dependence I observed. This holds even under float64 manifold ops.
- **Eval-method gap is recipe-driven, not eval-driven.** The 11 PPL gap between stochastic 50-batch evaluation (Preflight 1) and full-split ordered evaluation (Preflight 1b) was confirmed to come from training recipe differences (float64 manifold ops, final-20% cosine decay), not from the evaluation method itself.

Full pre-registration with all locked decisions, outcome criteria, and Amendment A.1.2: [PRE_REGISTRATION.md](PRE_REGISTRATION.md).

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
├── PRE_REGISTRATION.md       Experiment A pre-registration (main grid not executed)
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

**Foundational:**
- Nickel & Kiela (NeurIPS 2017) - [Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039). The starting point for hyperbolic representation learning.
- Nickel & Kiela (ICML 2018) - [Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry](https://arxiv.org/abs/1806.03417). Switch to the Lorentz model.

**Hyperbolic transformers and LLMs:**
- Chen et al. (ACL 2022) - [Fully Hyperbolic Neural Networks](https://arxiv.org/abs/2105.14686). First Lorentz-native architecture.
- Yang et al. (KDD 2024) - [Hypformer: Exploring Efficient Transformer Fully in Hyperbolic Space](https://arxiv.org/abs/2407.01290). Module library (HTC/HRC) for fully hyperbolic transformers.
- Yang et al. (2024) - [Hyperbolic Fine-tuning for Large Language Models (HypLoRA)](https://arxiv.org/abs/2410.04010). Empirical δ-hyperbolicity measurements of LLM token embeddings; analysis of the exp/log map cancellation problem.
- He et al. (2025) - [HyperCore: The Core Framework for Building Hyperbolic Foundation Models](https://arxiv.org/abs/2504.08912). Reference framework HELM is built on.
- He et al. (NeurIPS 2025) - [HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts](https://arxiv.org/abs/2505.24722). **Closest prior work.** First fully hyperbolic decoder-only LLMs at billion-parameter scale; uses fixed curvature in attention, mixture-of-curvature in FFN.

**Tooling:**
- [Geoopt](https://github.com/geoopt/geoopt) - Riemannian optimisation for PyTorch.

---

## Closing note

This repository is a completed exploration. I am a solo engineer, not an academic, and the hyperbolic LLM research programme is moving quickly in a well-resourced lab (Yang, He, Ying et al. at Yale and CUHK) publishing at NeurIPS/KDD cadence. Competing directly on their central research directions is not well-matched to my constraints.

The specific architectural gap this work targeted - per-head learnable curvature in attention, and its interpretability - is a narrow axis that HELM did not explore. The findings here are real (layer stratification on prose, scale-dependent stability wall, float64-as-overfitting on prose) but do not translate into the order-of-magnitude efficiency improvement the original research question demanded.

I may revisit this if one of the systems-level openings (quantisation, on-device inference for hyperbolic operations) becomes more tractable. Until then, these notes and the code exist so someone else doesn't have to repeat the scouting phase. Issues and questions welcome.
