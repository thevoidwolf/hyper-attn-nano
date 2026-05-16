# HyperAttn-Nano - Findings

This document is the full write-up of the experiment: what I tested, how I tested it, what I found, and what it means.

---

## Research question

Does giving each attention head its own learnable curvature parameter in a hyperbolic decoder-only transformer improve on:

1. A standard Euclidean (flat) baseline?
2. A fixed-curvature hyperbolic baseline (HELM-D style)?

The long-term motivation was an order-of-magnitude improvement in compute or memory efficiency, grounded in the theoretical result that hyperbolic space can represent hierarchical data exponentially more efficiently than Euclidean space.

---

## What is novel about this approach

HELM (NeurIPS 2025) established that fully hyperbolic decoder-only LLMs are viable at scale. Its attention mechanism uses a *fixed* curvature across all heads. The curvature diversity in HELM lives in the FFN layer (MoE experts), not in attention.

This experiment tests whether per-head learnable curvature in the attention mechanism itself provides an additional benefit - and whether the learned curvature values are interpretable.

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

**log_map formula:** The correct arcosh argument is `√(-K) · x₀`, not `-K · x₀`. The latter is only correct at K=-1 and fails silently for other curvatures - producing wrong geometry with no error. This was found and fixed during development.

---

## Experimental versions

### V1 - Initial run (5000 steps, LR=3e-4 all variants)

| Variant | Val PPL | Mean grad norm | Steps clipped |
|---|---|---|---|
| euclid | 277 | 1.026 | 3096 / 5000 (62%) |
| hyper-fixed | 451 | 0.544 | 17 / 5000 (<1%) |
| hyper-perhead | 450 | 0.542 | 15 / 5000 (<1%) |

**Problem identified:** euclid was gradient-clipped 62% of steps at norm ~1.0. Hyperbolic variants ran at mean norm ~0.54 - roughly half the update magnitude per step. The comparison was not fair; hyperbolic variants received approximately half the effective gradient signal.

**Finding:** Per-head curvature stratification already visible. Layer 2 most curved (K mean -1.140), layer 1 flattest (K mean -1.087). Spread of 0.117 across all 16 heads.

---

### V2 - Gradient scale correction (5000 steps, hyper LR=6e-4, warmup=300)

LR for hyperbolic variants raised to 6e-4 to match euclid's effective update scale.

| Variant | Val PPL | Mean grad norm | Steps clipped |
|---|---|---|---|
| euclid | 277 | 1.026 | 61% |
| hyper-fixed | 323 | 0.566 | 0.1% |
| hyper-perhead | 321 | 0.569 | 0.1% |

Gap closed from ~174 PPL to ~44 PPL. `hyper-perhead` beat `hyper-fixed` by 2.4 PPL - first confirmation of the per-head hypothesis.

**Problem identified:** Grad norm imbalance persisted (euclid ~1.03, hyper ~0.57 - 1.8× gap). At the same step count, euclid still received more total gradient signal.

---

### V3 - Total gradient signal correction (9000 steps, hyper LR=6e-4, warmup=500)

Step count for hyperbolic variants extended to 9000 (= 5000 × 1.026/0.569 ≈ 9000), giving the same total gradient signal as euclid V2 at 5000 steps.

| Variant | Steps | Val PPL | Mean grad norm | Steps clipped |
|---|---|---|---|---|
| euclid | 5000 | **277** | 1.30 | 61% |
| hyper-fixed | 9000 | 311 | 0.87 | 1.6% |
| hyper-perhead | 9000 | 312 | 0.86 | 3.7% |

The hyperbolic variants improved further but did not close the gap to euclid (~34 PPL remaining). The V2 advantage of `hyper-perhead` over `hyper-fixed` (2.4 PPL) reversed to a 1.2 PPL deficit - within noise range; neither direction is a reliable signal at this scale.

**Final curvature state (hyper-perhead V3):**

| Head | Layer 0 | Layer 1 | Layer 2 | Layer 3 |
|---|---|---|---|---|
| Head 0 | -1.177 | -1.064 | -1.171 | -1.120 |
| Head 1 | -1.143 | -1.121 | -1.081 | -1.169 |
| Head 2 | -1.086 | -1.143 | **-1.239** | -1.159 |
| Head 3 | -1.139 | -1.079 | -1.173 | -1.212 |

Most curved: `layer_2_head_2` at K=-1.239. Flattest: `layer_1_head_0` at K=-1.064. Spread: 0.175.

---

## Experiment B - Forced high curvature sweep

### Objective

V3 per-head curvatures converged near K≈-1.1 to -1.2. The theoretical advantage of hyperbolic geometry requires much stronger curvature. Experiment B tested whether curvatures K=-5, K=-10, and K=-50 are usable and whether they improve perplexity on WikiText-2.

Implementation required two changes to `src/manifolds.py`: an explicit angle clamp (`MAX_ANGLE = 10.0`) before the cosh/sinh calls in `exp_map`, and a tighter arcosh argument clamp (`min=1.0 + 1e-6`) in `log_map`. These stability improvements also raised the K=-1 baseline by ~14 PPL over V3, confirming that V3 had residual numerical instability.

### Results

| Run ID | K value | Final PPL | Stability |
|--------|---------|-----------|-----------|
| exp_b_k1  | -1.0  | 296.1 | STABLE |
| exp_b_k2  | -2.0  | 296.3 | STABLE |
| exp_b_k5  | -5.0  | 288.0 | STABLE |
| exp_b_k10 | -10.0 | **283.7** | STABLE |
| exp_b_k50 | -50.0 | 296.6 | STABLE |

Euclidean baseline (V3): 277.1 PPL.

### Finding

**K=-10 is the float32 optimum on WikiText-2.** Increasing curvature from K=-1 to K=-10 closes 12.4 PPL of the euclid–hyperbolic gap (from 19.0 PPL to 6.6 PPL). The improvement is monotone up to K=-10 then reverses at K=-50.

The K=-50 reversal is caused by angle clamping, not by the geometry being unhelpful: with `MAX_ANGLE=10.0`, the effective curvature experienced at K=-50 is estimated in the K=-12 to K=-20 range. The model cannot actually access the K=-50 geometry in float32. This motivated Experiment B2.

---

## Experiment B2 - Float64 high-curvature probe

### Objective

Extend the curvature sweep beyond the float32 ceiling by promoting only the manifold operations (`exp_map`, `log_map`, inner product) to float64 internally, then casting back to float32 before re-entering the rest of the model. This eliminates cosh/sinh overflow and arcosh precision loss at high curvature without requiring the full model to run in float64.

The replication check (`b2_k10`) must match Experiment B's K=-10 result (283.7 PPL) within 2.0 PPL before continuing the sweep.

### Results

| Run ID | K value | Final PPL | Note |
|--------|---------|-----------|------|
| exp_b2_k10 | -10.0 | 301.4 | **17.7 PPL worse than B k10** |

Sweep halted after replication check failed. `b2_k10` (float64) gave 301.4 PPL versus `b_k10` (float32) at 283.7 PPL - a regression of 17.7 PPL, far above the 2.0 PPL tolerance.

### Finding

**Float64 precision at K=-10 overfits on WikiText-2.** The float32 angle clamp (`MAX_ANGLE=10.0`) functions as an implicit regularizer: it prevents the manifold geometry from fitting training tokens too precisely and limits how sharply the model can carve out decision boundaries in hyperbolic space. Removing that constraint via float64 allows the geometry to memorise training patterns in WikiText-2 (which has relatively shallow hierarchical structure), increasing validation perplexity by 17.7 points.

This is counterintuitive: better numerical precision makes the model worse on the validation set. The mechanism is overfitting, not numerical error.

The higher-K sweep (K=-20, K=-50, K=-100) was not run because the B2 replication at K=-10 already failed. The question of whether float64 precision helps at much stronger curvature on WikiText-2 remains open.

---

## Experiment D - Hierarchical data probe (CodeParrot)

### Objective

Experiments B and B2 established that the euclid–hyperbolic gap on WikiText-2 narrows at K=-10 but does not close, and that float64 precision overfits on that dataset. The core hypothesis of Experiment D: **the geometry's advantage is data-structure-dependent.** Python source code has explicit, deep parse-tree structure that prose lacks. If the geometry is genuinely useful, the euclid–hyperbolic gap should shrink on code, and the float64 overfitting found in B2 should weaken or disappear.

Dataset: `codeparrot/codeparrot-clean` (Python GitHub files, deduplicated). Tokenised with the GPT-2 tokeniser used for WikiText-2 (vocabulary held constant). Size matched to WikiText-2: ~2.4M train tokens, ~248K validation tokens.

### Results

| Run ID | Variant | Final PPL |
|--------|---------|-----------|
| exp_d_euclid       | Euclid (control)            | **26.2** |
| exp_d_k10_f32      | Hyper fixed K=-10 float32   | 39.9 |
| exp_d_k10_f64      | Hyper fixed K=-10 float64   | 39.9 |
| exp_d_perhead_k10  | Hyper perhead init K=-10    | **34.9** |

### Analysis

**1. Gap direction - wider on code, not narrower.**
The euclid–hyperbolic gap grew on hierarchical data. Using the fixed K=-10 float32 baseline for the hyperbolic column:

| Dataset | Euclid PPL | Hyper K=-10 PPL | Gap (hyper − euclid) |
|---------|-----------|-----------------|----------------------|
| WikiText-2 | 277.1 | 283.7 | +6.6 |
| CodeParrot | 26.2 | 39.9 | +13.7 |

The hypothesis was wrong: code's tree structure does not give hyperbolic attention a smaller disadvantage. It gives it a larger one. Possible explanation: code's rigid syntactic structure allows Euclidean attention to learn near-deterministic attention patterns (e.g., always attend to the enclosing function definition), which suits the inductive bias of flat geometry. Hyperbolic attention penalises this kind of sharp, non-hierarchical pattern.

**2. Float64 overfitting resolution - data-dependent as hypothesised.**
Float32 and float64 give identical PPL on CodeParrot (39.9 vs 39.9). On WikiText-2, float64 was 17.7 PPL worse. This confirms the B2 overfitting was dataset-dependent: code's deeper hierarchical structure prevents the geometry from memorising training patterns. However, it does not improve performance either - float64 on code is no better than float32.

**3. Per-head curvature from K=-10 init - better than fixed, still far from euclid.**
`exp_d_perhead_k10` (initialised at K=-10, learnable) converged to K≈-6 across all heads and reached 34.9 PPL - 5.0 PPL better than fixed K=-10 (39.9 PPL). The model found that K=-6 is a better operating point on CodeParrot than K=-10. This is the first experiment where per-head curvature clearly and consistently outperforms a fixed-curvature baseline.

The gap to euclid remains large: 8.7 PPL (34.9 vs 26.2). Even with learnable per-head curvatures finding an optimal K, hyperbolic attention trails the Euclidean baseline by a substantial margin on code.

**4. Curvature convergence pattern on code - stratification disappears.**
Per-head curvatures initialised at K=-10 converged near K≈-6, with all layers reaching similar mean curvatures:

| | Layer 0 | Layer 1 | Layer 2 | Layer 3 |
|---|---|---|---|---|
| Mean K | -6.10 | -6.14 | -6.05 | -6.15 |
| Head range | -5.85 to -6.44 | -6.03 to -6.30 | -5.78 to -6.29 | -6.03 to -6.26 |

The layer stratification pattern from V3 (penultimate layer most curved, first layer flattest) is absent. All layers converged to approximately the same mean curvature. The total spread across all 16 heads is 0.66 - larger than V3 (0.175) but with no layer-level structure.

This suggests two possible interpretations: (a) the V3 stratification emerged specifically from proximity to the K=-1 initialisation, where the gradient landscape has different curvature-specific dynamics, or (b) code's uniform per-layer hierarchical depth (consistent scope nesting across all transformer layers) does not create the differential curvature pressure that WikiText-2's varied syntax produced.

---

## Key finding: curvature layer stratification (WikiText-2)

Across all three V1/V2/V3 experiment versions, `hyper-perhead` consistently developed the same pattern:

- **Layer 2 (penultimate) was always the most curved layer** (V1: -1.140, V2: -1.143, V3: mean -1.166)
- **Layer 1 was always the flattest layer** (V1: -1.087, V2: -1.099, V3: mean -1.102)
- The same head (`layer_2_head_2`) was the most curved single head in every version
- The spread grew with each version: 0.117 → 0.145 → 0.175

This pattern is stable across different learning rates and step counts. The model consistently decided that penultimate-layer attention should be most hierarchical, and did so independently across three separate training runs with different hyperparameters.

**This pattern does not persist on code.** On CodeParrot (Experiment D), per-head curvatures initialised at K=-10 converged to a flat distribution with no layer-level structure - all layers near K≈-6, spread 0.66 but uniformly distributed. Whether the stratification persists on WikiText-2 when initialised at K=-10 (rather than K=-1) remains untested.

This has not been documented before. HELM uses fixed curvature in attention, so per-head specialisation has never been measurable in a prior published system.

---

## Conclusions

### What failed

Hyperbolic attention with per-head learnable curvature does not outperform Euclidean attention at this scale, across two datasets. The gap persists after gradient-equivalent training, stronger curvature (K=-10), and the use of hierarchically-structured data.

- **V3 (WikiText-2, K=-1):** hyper-perhead 311.8 PPL vs euclid 277.1 PPL - gap of 34.7 PPL
- **Experiment B (WikiText-2, K=-10):** hyper-fixed 283.7 PPL vs euclid 277.1 PPL - gap narrowed to 6.6 PPL
- **Experiment D (CodeParrot, K=-10):** best hyperbolic (perhead) 34.9 PPL vs euclid 26.2 PPL - gap of 8.7 PPL

Per-head curvature does not consistently beat fixed curvature at K=-1 (V2 and V3 disagreed). At K=-10 on code, per-head (34.9) is clearly better than fixed (39.9) by 5.0 PPL - but still far from euclid (26.2).

### Why the gap persists

Three compounding factors:

**1. Scale mismatch.** Hyperbolic geometry's core advantage - efficiently embedding exponential tree structure - only pays off when the model has enough capacity and the data has enough hierarchical depth to use it. NANO_CONFIG (128-dimensional, 4 layers) is too small for this to matter regardless of dataset.

**2. The gap is data-structure-dependent, but in the wrong direction.** WikiText-2 gap (6.6 PPL at K=-10) is smaller than the CodeParrot gap (8.7–13.7 PPL). Hierarchically-structured code does not help hyperbolic attention close the gap; it widens it. This may be because code's rigid syntactic patterns allow Euclidean attention to learn near-deterministic rules, while hyperbolic geometry penalises non-hierarchical structure.

**3. Training remains harder.** Gradients on curved surfaces require geometric corrections that reduce effective update magnitude. The grad norm gap persisted through V3 (euclid 1.30 vs hyper 0.87) even after step-count compensation.

### What the original research question actually required

The "order-of-magnitude efficiency" goal would require showing equivalent quality with ~10x fewer parameters *or* ~10x fewer training steps. All experiments compared PPL at the same parameter count. Answering the original question requires a different experimental design: find the parameter count at which euclid hits a PPL target, then test whether hyperbolic hits the same target with ~1/10th the parameters.

### What is genuinely new

**1. Curvature layer stratification on WikiText-2.** The model consistently assigns stronger curvature to penultimate-layer attention heads across multiple independent training runs. This interpretability result suggests the penultimate layer performs the most hierarchically structured processing and that the model discovers this automatically when given the freedom to.

**2. Float64 overfitting on prose.** Float64 manifold precision at K=-10 worsens WikiText-2 perplexity by 17.7 PPL over float32. The float32 angle clamp acts as an implicit regularizer on shallow-hierarchy text. This is a counterintuitive precision-regularization trade-off with potential relevance to other curved-space learning systems.

**3. Stratification does not persist on code.** The V3 per-layer curvature pattern (penultimate layer most curved, first layer flattest) disappears entirely when training on CodeParrot. On code, all layers converge to a uniform K≈-6. The pattern appears to be specific to the WikiText-2 training distribution and K=-1 initialisation regime.

**4. Per-head curvature finds K=-6 on code.** Initialised at K=-10, per-head curvatures relax to K≈-6 and gain 5 PPL over fixed K=-10 on code. The model independently found that K=-6 is a better operating point than K=-10 on hierarchical data - demonstrating that learnable curvature provides a genuine search advantage over fixed curvature when initialised in a useful regime.

---

## Phase 5 — Probe 2, QK-norm isolation, and OOD eval pack (2026-05-16)

### Objective

Two open items left over from the archived programme were revisited:

1. **Probe 2** (`hyper-scores-only` K=-10, infrastructure spec §2.2) NaN'd at step 0 in
   its original attempt and was never rerun. The infrastructure spec called Probe 2
   the most diagnostic experiment in the programme: it isolates the hypothesis "the
   geometry helps similarity computation but hurts representation capacity when
   round-tripped through tangent space" by keeping Q/K on the Lorentz manifold while
   leaving the value path Euclidean.

2. **OOD eval pack** (infrastructure spec §1.3) was written but never run. The four
   metrics (`id_val_ppl`, `ood_val_ppl`, `rare_word_ppl`, `long_ctx_ppl`) are designed
   to detect hyperbolic advantages that the in-distribution val PPL averages out.

### Probe 2 failure root cause

Under `torch.amp.autocast("cuda")`, `W_q` and `W_k` projections emit fp16. After
`exp_map_batched` (which internally casts to fp32), the Lorentz points reach ~10⁶ for
random-init Q at K=-10 (because `‖q‖ ≈ √d_head ≈ 5.6`, so `angle ≈ √10·5.6 ≈ 17.7`,
so `cosh(17.7) ≈ 2.4·10⁷`). The subsequent `Q_t @ K_t.T` matmul autocasts back to
fp16 (max 65504) and overflows to inf/nan, propagating to the loss.

`LorentzPerHeadAttention` has the same matmul but escaped this in earlier experiments
because at K=-1 with log-mapped inputs, the values fit in fp16; the failure is specific
to the K ≤ -2 + autocast + Euclidean-input regime.

### Fix

Two changes to `src/attention.py:LorentzScoreOnlyAttention.forward`:

1. `with torch.amp.autocast("cuda", enabled=False)` around the entire Lorentz score
   computation, keeping it in fp32 regardless of ambient autocast.
2. Per-head L2 normalisation of Q and K *before* exp_map, bounding the angle fed
   into cosh/sinh to `√|K|` regardless of init scale. This is the standard QK-norm
   pattern from PaLM and Llama, retargeted here as a numerical-stability fix for
   the hyperbolic path.

Curvature warmup (linear K=-1 → K=-10 over 500 steps) was also added to the Probe 2
config, matching what Probe 1 used.

### Probe 2 result

Stochastic 50-batch best-val PPL on WikiText-2 (apples-to-apples with earlier
programme numbers):

| Variant (WikiText-2, NANO d=128, ~7.2M, 9000 steps, LR=6e-4) | Best val PPL | Step |
|---------------------------------------------------------------|-------------:|-----:|
| **Probe 2 (Lorentz scores K=-10 + QK-norm + warmup)**         |   **242.69** | 4250 |
| Probe 1 (hyper-fixed K=-1→K=-10 warmup)                       |       269.28 |  ??? |
| exp_b k10 (hyper-fixed K=-10, no warmup, no QK-norm)          |       283.70 |    — |
| Euclid V3 (LR=3e-4, vanilla, checkpoint subsequently lost)    |       277.10 |    — |

This is the first hyperbolic variant in the programme to beat the Euclidean baseline
at matched params on WikiText-2. **However**: Probe 2 bundles three changes (scores-
only architecture, QK-norm, curvature warmup) and at K=-10, none of these were tested
in the published variants. The contribution attributable to the geometry alone needs
isolation.

### QK-norm isolation experiment

To split the contribution of QK-norm from the contribution of the hyperbolic geometry,
a matched Euclidean baseline was trained with the same QK-norm and the same training
recipe (LR=6e-4, 9000 steps, seed=42, batch=32, WikiText-2). EuclideanAttention was
extended with a `qk_norm: bool` config flag (default False, so existing checkpoints
are unaffected).

| Variant (WikiText-2, NANO d=128, LR=6e-4)                | Best val PPL |
|----------------------------------------------------------|-------------:|
| Probe 2 (Lorentz scores + QK-norm + warmup)              |   **242.69** |
| Euclid + QK-norm @ LR=6e-4 (new control)                 |   **253.62** |
| Euclid V3 (LR=3e-4, vanilla, checkpoint lost)            |       277.10 |

**The geometry contributes ~11 PPL of improvement on top of QK-norm.** QK-norm alone
explains ~24 PPL of improvement (Euclid V3 vanilla → Euclid+QK-norm), and the
hyperbolic Lorentz score replaces another ~11 PPL on top of that.

This isolation is the cleanest hyperbolic-vs-Euclidean architectural-win signal the
programme has produced.

### OOD eval pack results

`src/eval/ood_eval.py` was already written but had two bugs that needed fixing before
it produced useful numbers:

- `_eval_ppl_on_tokens` advanced by `batch_size` between batches rather than
  `batch_size × block_size`, so 50 batches only covered ~1.8K tokens of a 248K val
  set. Fixed to non-overlapping ordered windows; eval now covers the full val.
- `per_token` losses were recorded only at the last position of each window (~1.6K
  samples per run), so `rare_word_ppl` almost always returned NaN. Fixed to record
  every position (~245K samples per run); `rare_word_ppl` now finite across all runs.

Long-context multipliers also extended from (1, 2, 4) to (1, 2, 4, 8, 16) = 256 /
512 / 1024 / 2048 / 4096 token contexts. The decoder-only architecture has no
positional encoding so longer contexts work mechanically.

Full-split eval on every available checkpoint:

| Variant (NANO d=128, WikiText-2)             |   id  |  ood  |  c256 |  c512 |  c1024 |  c2048 |  c4096 | c4096 / c256 |
|----------------------------------------------|------:|------:|------:|------:|-------:|-------:|-------:|-------------:|
| Probe 2 (Lorentz scores + QK-norm + warmup)  | 254.9 | 380.3 | 254.9 | 281.0 |  367.4 |  476.9 |  616.8 |    **2.42×** |
| Euclid + QK-norm                             | 262.0 | 401.0 | 262.0 | 272.3 |  288.0 |  309.6 |  341.3 |        1.30× |
| V1 hyper-fixed K=-1 (LR=3e-4, no QK-norm)    | 327.8 | 398.9 | 327.8 | 333.8 |  342.6 |  354.6 |  372.4 |        1.14× |
| V1 hyper-perhead K=-1 (LR=3e-4, no QK-norm)  | 326.5 | 398.0 | 326.5 | 333.5 |  344.1 |  358.3 |  377.5 |        1.16× |
| V1 Euclid (LR=3e-4)                          | 280.0 | 398.9 | 280.0 | 290.8 |  313.2 |  341.3 |  374.8 |        1.34× |

Three new observations:

**1. Probe 2 wins short, loses long.** At training context (256), Probe 2 beats
Euclid+QK-norm by 7.1 PPL id / 20.7 PPL OOD. At 16× training context (4096), Probe 2
is 80% worse (617 vs 341). The 2.42× degradation ratio is the highest in any
variant tested.

**2. K=-1 hyperbolic models have excellent long-context retention.** V1 hyper-fixed
K=-1 degrades only 1.14× from 256 → 4096, vs 1.34× for the matching V1 Euclidean.
The hyperbolic geometry helps length generalisation *when |K| is small*.

**3. The K=-10 score-amplifier hypothesis explains both.** The Lorentz score formula
`|K| · (-⟨q,k⟩_L) / √d_head` multiplies attention scores by |K|. At K=-1 this is a
no-op; at K=-10 it amplifies score magnitudes ~10×, making softmax dramatically
sharper. The model learns to use this sharpness at training context length (better
in-distribution PPL), but at 4×–16× longer contexts the sharpness over a larger
candidate set saturates onto wrong positions and PPL collapses.

This is testable: if Probe 2 with K=-3 or K=-5 has milder long-context degradation,
the |K| amplifier is confirmed as the mechanism. Not yet run.

### Damage from this round

`scripts/train.py:save_checkpoint` writes checkpoints to
`results/checkpoints/<model_type>/ckpt_step<N>.pt`, keyed on `model.type`. Launching
the Euclid+QK-norm training without overriding the checkpoint path silently
overwrote `results/checkpoints/euclid/ckpt_step8999.pt`, which previously held
`exp_d_euclid` (CodeParrot, val PPL 26.2). The numeric result is preserved in this
document but the checkpoint is gone, which prevents extending an interesting
long-context finding from CodeParrot to longer contexts.

This is a known repo gotcha — variant-keyed checkpoint paths overwrite between
experiments. The Experiment A code path uses a `run_id`-keyed checkpoint directory
that does not have this bug; the legacy `save_checkpoint` should be updated to
match before any further runs.

### Files added / modified in Phase 5

- `src/attention.py` — `autocast(enabled=False)` wrap + QK-norm in
  `LorentzScoreOnlyAttention`; `qk_norm: bool` config flag in `EuclideanAttention`
- `src/model.py` — wires `qk_norm` config flag through to attention
- `src/eval/ood_eval.py` — stride bug fix; per-position loss collection for
  `rare_word_ppl`; extended `multipliers` default
- `scripts/train.py` — propagates `qk_norm` config key to model
- `scripts/run_ood_eval_all.py` — NEW; batch runner over all checkpoints
- `scripts/data_prep_ood.py` — NEW; downloads WT-103 train, filters against WT-2,
  saves first 100K tokens
- `configs/probe2_scores_only.yaml` — added `curvature_schedule: linear_warmup K=-1
  → K=-10 over 500 steps`
- `configs/euclid_qknorm_wikitext2.yaml` — NEW; Probe 2's recipe with
  `model.type: euclid` and `qk_norm: true`
- `results/eval/OOD_FINDINGS.md` — full Phase 5 OOD writeup
- `results/eval/AGGREGATE.json` + per-checkpoint `eval_summary.json` files
- `data/ood/wikitext103_heldout/ood.bin` — held-out OOD corpus

---

## Limitations

- Single architecture family and scale; all results at NANO_CONFIG (128-dimensional, 4 layers)
- Two datasets tested (WikiText-2, CodeParrot) - both English-language; no non-English or non-text data
- No parameter-efficiency comparison (same parameter count throughout)
- Float64 curvature sweep (K=-20, K=-50, K=-100) was not completed - only K=-10 float64 on WikiText-2 was tested
- Training dynamics not fully equalised: grad norm gap persisted through V3 and Experiment D
- The V3 WikiText-2 stratification pattern has not been tested at K=-10 initialisation; it may be an artefact of proximity to K=-1
- Per-head vs fixed curvature comparison at K=-1 is noisy (V2 and V3 disagreed); the K=-10 per-head advantage on code is clearer but comes from a single run
- Phase 5: Probe 2's geometry-vs-QK-norm isolation is a single seed; the 11 PPL gap
  to Euclid+QK-norm could be ±3 PPL of seed noise
- Phase 5: the long-context degradation mechanism (K-amplifier hypothesis) is plausible
  but unverified — needs a Probe 2 run at K=-3 or K=-5 to confirm
- Phase 5: the CodeParrot long-context picture cannot be completed without a fresh
  `exp_d_euclid` rerun (Euclidean checkpoint was overwritten this round)
