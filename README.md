# HyperAttn-Nano

A minimal proof-of-concept decoder-only language model with **per-head learnable curvature in hyperbolic attention**. Built to test whether giving each attention head its own curvature parameter improves on fixed-curvature hyperbolic attention (the approach used in [HELM](https://arxiv.org/abs/2505.24722)).

**Short answer: it does not outperform a Euclidean baseline at this scale.** But the experiment produced a novel and reproducible finding about how curvature self-organises across layers. See [FINDINGS.md](FINDINGS.md) for the full write-up.

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

Three experiment versions, each correcting a training fairness issue identified in the previous one.

| Version | What changed | euclid PPL | hyper-fixed PPL | hyper-perhead PPL |
|---|---|---|---|---|
| V1 (5k steps, LR=3e-4 all) | Baseline | 277 | 451 | 450 |
| V2 (5k steps, hyper LR=6e-4) | Equalised gradient scale | 277 | 323 | 321 |
| V3 (9k steps, hyper LR=6e-4) | Equalised total gradient signal | 277 | 311 | 312 |

Even after gradient-equivalent training, the hyperbolic variants trail the Euclidean baseline by ~34 perplexity points. See [FINDINGS.md](FINDINGS.md) for the explanation.

---

## Key finding: curvature layer stratification

The most interesting result is not the perplexity comparison — it's what the per-head curvature learned. Across all three experiment versions, `hyper-perhead` consistently developed a stable layer-wise pattern:

- **Layer 2 (penultimate) was always most curved** — K reaching -1.24 by V3
- **Layer 1 was always flattest** — K staying close to -1.06
- The spread across all heads grew with each version: 0.117 (V1) → 0.145 (V2) → 0.175 (V3)
- The same head (`layer_2_head_2`) was the most curved head in every version

This stratification pattern has not been documented in prior literature. HELM has fixed curvature in attention, so this kind of per-head specialisation has not previously been observable.

---

## Repo structure

```
hyper-attn-nano/
├── src/
│   ├── manifolds.py        Lorentz manifold ops (exp_map, log_map, inner product)
│   ├── attention.py        LorentzPerHeadAttention + EuclideanAttention
│   ├── blocks.py           LorentzRMSNorm, LorentzFFN, decoder block
│   └── model.py            HyperAttnNano + GPTNano baseline
├── scripts/
│   ├── data_prep.py        Download + tokenise WikiText-2
│   ├── train.py            Training loop (all variants)
│   ├── compare.py          PPL comparison plots across versions
│   └── viz_curvature.py    Per-head curvature evolution + heatmap plots
├── configs/
│   ├── nano_euclid.yaml              V1 config
│   ├── nano_hyper_fixed.yaml         V1 config
│   ├── nano_hyper_perhead.yaml       V1 config
│   ├── nano_*_v2.yaml                V2 configs (hyper LR raised to 6e-4)
│   └── nano_*_v3.yaml                V3 configs (9000 steps)
├── results/
│   ├── logs/{v1,v2,v3}/              Training logs (loss, PPL, curvatures per step)
│   └── plots/                        Generated plots
├── tests/                            Unit tests for all src/ modules
├── setup/smoke_test.py               Quick environment check
├── requirements.txt
└── FINDINGS.md                       Full experimental write-up
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

# Prepare data (downloads WikiText-2, ~2 min)
python scripts/data_prep.py --dataset wikitext2 --out data/wikitext2

# Run V3 training (adjust config for the variant you want)
python scripts/train.py --config configs/nano_euclid.yaml
python scripts/train.py --config configs/nano_hyper_fixed_v3.yaml
python scripts/train.py --config configs/nano_hyper_perhead_v3.yaml

# Generate plots from existing logs
python scripts/compare.py
python scripts/viz_curvature.py --version v3
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
