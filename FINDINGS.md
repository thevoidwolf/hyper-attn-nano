# HyperAttn-Nano — Findings

This document is the full write-up of the experiment: what we tested, how we tested it, what we found, and what it means.

---

## Research question

Does giving each attention head its own learnable curvature parameter in a hyperbolic decoder-only transformer improve on:

1. A standard Euclidean (flat) baseline?
2. A fixed-curvature hyperbolic baseline (HELM-D style)?

The long-term motivation was an order-of-magnitude improvement in compute or memory efficiency, grounded in the theoretical result that hyperbolic space can represent hierarchical data exponentially more efficiently than Euclidean space.

---

## What is novel about this approach

HELM (NeurIPS 2025) established that fully hyperbolic decoder-only LLMs are viable at scale. Its attention mechanism uses a *fixed* curvature across all heads. The curvature diversity in HELM lives in the FFN layer (MoE experts), not in attention.

This experiment tests whether per-head learnable curvature in the attention mechanism itself provides an additional benefit — and whether the learned curvature values are interpretable.

---

## Architecture

All three variants share the same decoder-only transformer structure (NANO_CONFIG: d_model=128, 4 layers, 4 heads, d_ff=512, max_seq_len=256, ~5M parameters). The only difference is the attention and normalisation implementation.

**euclid:** Standard scaled dot-product attention. Residuals in Euclidean space throughout.

**hyper-fixed / hyper-perhead:** Tokens are lifted to the Lorentz hyperboloid via `exp_map` after embedding. Each decoder block operates:
- `LorentzRMSNorm`: log_map → RMSNorm → exp_map
- `LorentzPerHeadAttention`: attention scores use the Lorentz inner product `⟨q,k⟩_L = -q₀k₀ + Σqᵢkᵢ`, scaled by `|K|`
- `LorentzFFN`: log_map → Linear → GELU → Linear → exp_map
- Residuals: `exp_map(log_map(x) + log_map(attn_out))`

In `hyper-perhead`, each head has its own scalar `log_abs_K` parameter initialised to 0 (→ K=-1). Curvature is recovered as `K = -exp(log_abs_K)`, keeping it strictly negative and unconstrained during optimisation.

### Key implementation notes

**Float32 stability:** The angle fed into cosh/sinh is `√(-K) · ‖v‖`. Float32 loses the identity `cosh² − sinh² = 1` for angles above ~4, causing manifold constraint violations. Fixed by initialising `LorentzRMSNorm` scale to `1/√d_model` (not ones) and using `std=0.02` embedding initialisation.

**log_map formula:** The correct arcosh argument is `√(-K) · x₀`, not `-K · x₀`. The latter is only correct at K=-1 and fails silently for other curvatures — producing wrong geometry with no error. This was found and fixed during development.

---

## Experimental versions

### V1 — Initial run (5000 steps, LR=3e-4 all variants)

| Variant | Val PPL | Mean grad norm | Steps clipped |
|---|---|---|---|
| euclid | 277 | 1.026 | 3096 / 5000 (62%) |
| hyper-fixed | 451 | 0.544 | 17 / 5000 (<1%) |
| hyper-perhead | 450 | 0.542 | 15 / 5000 (<1%) |

**Problem identified:** euclid was gradient-clipped 62% of steps at norm ~1.0. Hyperbolic variants ran at mean norm ~0.54 — roughly half the update magnitude per step. The comparison was not fair; hyperbolic variants received approximately half the effective gradient signal.

**Finding:** Per-head curvature stratification already visible. Layer 2 most curved (K mean -1.140), layer 1 flattest (K mean -1.087). Spread of 0.117 across all 16 heads.

---

### V2 — Gradient scale correction (5000 steps, hyper LR=6e-4, warmup=300)

LR for hyperbolic variants raised to 6e-4 to match euclid's effective update scale.

| Variant | Val PPL | Mean grad norm | Steps clipped |
|---|---|---|---|
| euclid | 277 | 1.026 | 61% |
| hyper-fixed | 323 | 0.566 | 0.1% |
| hyper-perhead | 321 | 0.569 | 0.1% |

Gap closed from ~174 PPL to ~44 PPL. `hyper-perhead` beat `hyper-fixed` by 2.4 PPL — first confirmation of the per-head hypothesis.

**Problem identified:** Grad norm imbalance persisted (euclid ~1.03, hyper ~0.57 — 1.8× gap). At the same step count, euclid still received more total gradient signal.

---

### V3 — Total gradient signal correction (9000 steps, hyper LR=6e-4, warmup=500)

Step count for hyperbolic variants extended to 9000 (= 5000 × 1.026/0.569 ≈ 9000), giving the same total gradient signal as euclid V2 at 5000 steps.

| Variant | Steps | Val PPL | Mean grad norm | Steps clipped |
|---|---|---|---|---|
| euclid | 5000 | **277** | 1.30 | 61% |
| hyper-fixed | 9000 | 311 | 0.87 | 1.6% |
| hyper-perhead | 9000 | 312 | 0.86 | 3.7% |

The hyperbolic variants improved further but did not close the gap to euclid (~34 PPL remaining). The V2 advantage of `hyper-perhead` over `hyper-fixed` (2.4 PPL) reversed to a 1.2 PPL deficit — within noise range; neither direction is a reliable signal at this scale.

**Final curvature state (hyper-perhead V3):**

| Head | Layer 0 | Layer 1 | Layer 2 | Layer 3 |
|---|---|---|---|---|
| Head 0 | -1.177 | -1.064 | -1.171 | -1.120 |
| Head 1 | -1.143 | -1.121 | -1.081 | -1.169 |
| Head 2 | -1.086 | -1.143 | **-1.239** | -1.159 |
| Head 3 | -1.139 | -1.079 | -1.173 | -1.212 |

Most curved: `layer_2_head_2` at K=-1.239. Flattest: `layer_1_head_0` at K=-1.064. Spread: 0.175.

---

## Key finding: curvature layer stratification

Across all three experiment versions, `hyper-perhead` consistently developed the same pattern:

- **Layer 2 (penultimate) was always the most curved layer** (V1: -1.140, V2: -1.143, V3: mean -1.166)
- **Layer 1 was always the flattest layer** (V1: -1.087, V2: -1.099, V3: mean -1.102)
- The same head (`layer_2_head_2`) was the most curved single head in every version
- The spread grew with each version: 0.117 → 0.145 → 0.175

This pattern is stable across different learning rates and step counts. The model consistently decided that penultimate-layer attention should be most hierarchical, and did so independently across three separate training runs with different hyperparameters.

This has not been documented before. HELM uses fixed curvature in attention, so per-head specialisation has never been measurable in a prior published system.

---

## Conclusions

### What failed

Hyperbolic attention with per-head learnable curvature does not outperform Euclidean attention at this scale, even with gradient-equivalent training. The gap is ~34 PPL after V3, and the trend across versions is convergent but not crossing.

Per-head curvature does not consistently beat fixed curvature (-1.0). The V2 advantage (2.4 PPL) reversed in V3 (1.2 PPL in the other direction). The difference is within noise at NANO_CONFIG scale.

### Why

Three compounding factors:

**1. Scale mismatch.** Hyperbolic geometry's core advantage — efficiently embedding exponential tree structure — only pays off when the model has enough capacity and the data has enough hierarchical depth to use it. NANO_CONFIG (128-dimensional, 4 layers) is too small for this to matter.

**2. Text is not a pure tree.** Language has hierarchical structure but also many non-tree relationships: coreference, long-range dependencies, world knowledge. Euclidean attention handles the full mix without constraint; hyperbolic attention is optimised for a shape that text only partially has.

**3. Training is harder.** Gradients on curved surfaces require geometric corrections that reduce effective update magnitude. Even after the step-count compensation in V3, the grad norm gap persisted (euclid 1.30 vs hyper 0.87). The optimizer never fully equalised.

### What the original research question actually required

The "order-of-magnitude efficiency" goal would require showing equivalent quality with ~10x fewer parameters *or* ~10x fewer training steps. This experiment compared PPL at the same parameter count — it could establish whether hyperbolic attention is better at equal scale, but not whether it achieves equivalent quality at smaller scale. Answering the original question requires a different experimental design: find the parameter count at which euclid hits a PPL target, then test whether hyperbolic hits the same target with ~1/10th the parameters.

### What is genuinely new

The curvature stratification finding. The model consistently assigns stronger curvature to penultimate-layer attention heads across multiple independent training runs. This is an interpretability result: it suggests the penultimate layer performs the most hierarchically structured processing, and that the model can discover this automatically when given the freedom to.

Whether this pattern would persist at larger scale, or whether some heads would push toward K=-2 or K=-3 (as would be expected if the geometry were truly useful), remains an open question.

---

## Limitations

- Single dataset (WikiText-2), single architecture family, single scale
- No parameter-efficiency comparison (same parameter count throughout)
- NANO_CONFIG is too small to expect hyperbolic geometry advantages
- Grad norm gap persisted through V3 — training dynamics are not fully understood
- Per-head vs fixed curvature comparison is noisy at this scale (V2 and V3 disagreed)
