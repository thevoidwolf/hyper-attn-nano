# Experiment A — Pre-Registration

**Spec version:** A.1.1  
**Committed by:** Voidwolf  
**Date:** 2026-04-23  
**Status: LOCKED — do not edit after the first main-grid run begins.**

---

## 1. Research question

Can a hyperbolic decoder-only transformer match or exceed the best-val PPL of a *larger* Euclidean transformer on WikiText-2 at matched training budget?

---

## 2. Primary metric

Best-val PPL on the WikiText-2 validation split, averaged across 3 seeds per cell, reported with 95% confidence interval.

**Definition of "best val PPL":** The minimum val PPL observed at any eval step (every 250 steps) during training. This is not the final PPL and not the rolling mean. Do not change this definition post-hoc.

---

## 3. Primary efficiency axis

Non-embedding transformer-body parameters. The shared 50k-vocab embedding table is excluded from the efficiency metric as it is architecture-invariant.

---

## 4. Scale grid

Width-only scaling. L=4 layers and head_dim=32 held constant.

| Tag | d_model | d_ff  | n_heads | Non-emb body (approx) |
|-----|---------|-------|---------|----------------------|
| S1  | 64      | 256   | 2       | ~0.2M                |
| S2  | 128     | 512   | 4       | ~0.8M                |
| S3  | 256     | 1024  | 8       | ~3.1M                |
| S4  | 384     | 1536  | 12      | ~7.1M                |
| S5  | 512     | 2048  | 16      | ~12.6M               |

Exact non-embedding body param counts are computed from model state dicts and logged in each run's `summary.json`.

---

## 5. Variants

Two variants only:

- **Euclidean:** Standard multi-head attention, Pre-LN, GeLU FFN, weight-tied output head, float32 throughout. LR schedule: linear warmup + cosine decay with min_lr_ratio=0.1.

- **Hyperbolic (hyper-fixed):** Lorentz manifold attention, fixed K=-10 after warmup. Curvature warmup: linear K=-1 → K=-10 over first 500 steps. Float64 manifold ops (exp_map, log_map, Lorentz distance). Standard linear output head (weight-tied). LR schedule: linear warmup + cosine decay with additional cosine decay over the final 20% of training steps (single piecewise schedule).

`hyper-perhead` is not run in Experiment A.

---

## 6. Training configuration

- Dataset (primary): WikiText-2 (raw)
- Tokenizer: GPT-2 BPE, vocab 50257
- Sequence length: 256
- Batch size: 32
- Total steps: 9000
- Seeds: 42, 1337, 2718 (3 per cell)
- Gradient clipping: global norm 1.0
- Optimizer: AdamW (lr per variant, weight_decay=0.1, betas=(0.9, 0.95))
- Evaluation: full-split ordered non-overlapping pass over WikiText-2 val (every 250 steps)

---

## 7. Pre-registered decision criteria

Let `H(s)` = mean hyperbolic best-val PPL at scale `s` across 3 seeds, `E(s)` = mean Euclidean best-val PPL at scale `s`, each with 95% bootstrap CIs.

### Outcomes

- **Strong success (≥ ~4× efficiency):** There exists a scale `s` such that `H(s) ≤ E(s+2)` with non-overlapping 95% CIs (e.g. hyper-S2 vs Euclid-S4).
- **Modest success (~2–3× efficiency):** There exists a scale `s` such that `H(s) ≤ E(s+1)` with non-overlapping 95% CIs.
- **Tie (≈ 1× efficiency):** `H(s) ≈ E(s)` at some scale with overlapping CIs and no scale meets the success criteria above.
- **Falsified (< 1× efficiency):** `E(s) < H(s)` at every scale with non-overlapping CIs.

### Pre-committed analysis decisions

- **Plot scaling:** log-x (non-embedding body params), linear-y (best-val PPL). Do not change post-hoc.
- **CI method:** bootstrap with 10,000 resamples across the 3 seeds. Do not use normal approximation (n=3 is too small).
- **"Best val PPL":** minimum val PPL at any eval step. Do not change to final PPL or rolling mean post-hoc (both reported as secondary).
- **Outlier handling:** none. All 3 seeds are reported. If a single seed fails catastrophically (NaN loss), run a 4th seed (31415) and report both the 3-seed stat and the 4-seed-minus-failure stat. Do not silently drop a seed.

---

## 8. Measurement method notes (Addendum A.1.1)

### 8.1 Probe 1 reproduction vs main-grid measurement

Preflight 1 reproduces Probe 1's exact recipe: float32 manifold ops, stochastic 50-batch evaluation (~410k tokens sampled with replacement), no final-20% cosine decay. The success criterion (264–275 PPL) is evaluated against this stochastic number.

The main grid uses a deliberately different recipe: float64 manifold ops, full-split ordered evaluation, and the added final-20% cosine decay. The main-grid S2/hyperbolic result is therefore a **new measurement against a new recipe**, not a reproduction of 269.3. Preflight 1 validates that the architecture and data pipeline still produce Probe 1's number; Preflight 1b validates that the main-grid recipe produces a sensible number; the main grid itself produces the headline claim.

### 8.2 Evaluation method change

The stochastic 50-batch evaluation in the existing repo oversamples tokens (~410k sampled with replacement vs ~246k in WikiText-2 val) and produces noisier PPL estimates than a full-split ordered pass. All Experiment A results from Preflight 2 onward use full-split ordered evaluation. Headline numbers from the main grid are not directly comparable to Probe 1's 269.3.

### 8.3 Secondary finding to watch for

Probe 1's result was achieved with float32 manifold ops. The main grid effectively tests whether adding float64 on top of the warmup recipe helps, hurts, or is neutral at S2. If the main-grid S2/hyperbolic run produces a best-val PPL close to Preflight 1's full-split anchor, float64 contributed little at this scale — itself a finding worth reporting in `decision.md`.

---

## 9. What is out of scope for the headline claim

- Wallclock efficiency
- FLOPs efficiency  
- Data efficiency

All three are logged as secondary metrics and reported in `decision.md`, but the headline is parameter efficiency at matched best-val PPL.

---

## 10. Confirmation dataset

CodeParrot-clean, run only at the crossover scale identified on WikiText-2 after the main grid completes. CodeParrot results are secondary and do not change the WikiText-2 headline claim. A "did not generalise" result is valid and reported as-is.

---

*Pre-registration version: A.1.1 — locked at first main-grid run.*

---

## Amendment A.1.2 — Preflight 2 outcomes

Committed after Preflight 2 (§7.2) and Preflight 2b scouting runs completed,
prior to main-grid execution. This amendment locks in three decisions that
Preflight 2 surfaced.

### 1. Numerical stability wall at high d × high |K|

Preflight 2 revealed a hard numerical stability boundary previously unseen
in the programme. At S5 (d_model=512, ~12.6M non-embedding body params):

- K=-10 produced non-finite loss at step 2897 (~32% through training)
- K=-15 produced non-finite loss at step 2284 (~25% through training)
- K=-5 completed training normally and reached best-val PPL 229.9

At S1 (d_model=64), all three K values completed training and landed
within a 4.2 PPL band — indistinguishable at single-seed resolution.

Grad norms were well within the 1.0 clip at crash time (~0.9–1.1),
indicating the NaN originates in the forward-pass manifold operations,
not in gradient explosion. This is consistent with exp_map / log_map
overflow at high d combined with high |K|, even under float64.

This is reported as a first-class finding of Experiment A and is
expected to appear in `decision.md`: **optimal K decreases with model
scale, and the literature's default K=-1 is not merely suboptimal but
the opposite direction of the scale dependence we observe.**

### 2. Scale-dependent K for the main grid

Scale-dependent K is now mandatory for the main grid. The K=-10 lock
from §7.2 is superseded. The split boundary is determined by Preflight
2b, a single scouting run at S3 with K=-10, seed 42.

**Result of Preflight 2b:**

Preflight 2b (S3/K=-10/seed42) completed 9000 steps without NaN.
Best-val PPL 239.30 at step 5250. Peak VRAM 7.7 GB. No resume events.

**Main-grid K schedule (locked):**

| Scale | K     |
|-------|-------|
| S1    | -10   |
| S2    | -10   |
| S3    | -10   |
| S4    | -5    |
| S5    | -5    |

S4 adopts K=-5 conservatively. K=-10 is confirmed stable at S3 and
confirmed unstable at S5, but S4 was not directly tested. The
conservative call here reflects that one further NaN failure in the
main grid costs ~5 hours of wasted compute; the sub-optimal-K risk
is bounded and smaller. If S4 shows an anomalously weak result
relative to the S3→S5 trend, S4/K=-10 is a candidate follow-up.

The chosen schedule is locked at main-grid commit time and not
adjusted during or after the main grid. Seeds 1337 and 2718 at each
scale use the schedule established here.

### 3. Eval-method question — closed

§17.1 raised the possibility that the 11 PPL gap between Preflight 1
(stochastic 50-batch, 269.28 best-val) and Preflight 1b (full-split,
280.93 best-val) might be an eval-method artefact. To test this,
stochastic 50-batch eval was re-run on Preflight 1b's `best.pt`
(step 5250) across 5 RNG seeds.

Result: mean stochastic PPL = **283.20** (stdev 1.68, range
280.68–285.59). On this checkpoint the two eval methods agree to
within ~2 PPL. The gap between Preflight 1 and Preflight 1b is
therefore recipe-driven, not eval-driven.

The two candidate drivers (f64 manifold ops; final-20% cosine decay;
or their interaction) are not disentangled in Experiment A. This is
acknowledged as a limitation of the main-grid design and is flagged
for investigation in a follow-up experiment if warranted.

### 4. No change to the pre-registered success criteria

The outcome table in §12 is unchanged. Only the operating schedule
for K and the documentation of the eval-method finding are updated.
The headline metric remains best-val PPL on WikiText-2 val split,
measured under full-split ordered evaluation, averaged across 3 seeds
per cell with 95% bootstrap CI.