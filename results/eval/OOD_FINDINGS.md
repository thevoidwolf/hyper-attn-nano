# OOD Eval Pack — Findings (Round 2)

**Date:** 2026-05-16
**Eval method:** `_eval_ppl_on_tokens` was buggy in round 1 (stride=`batch_size`
within a 256-token window, so 50 batches covered only ~1.8K tokens of val).
Fixed to non-overlapping ordered windows. `rare_word_ppl` now records losses at
every position of each window (not just the last), giving meaningful per-token
samples. Long-context multipliers extended from (1, 2, 4) to (1, 2, 4, 8, 16) =
256 / 512 / 1024 / 2048 / 4096 tokens.
**OOD source:** WikiText-103 *train* split, lines filtered against WikiText-2
(`data/ood/wikitext103_heldout/ood.bin`, 100K tokens).

---

## Headline

Two new things since round 1:

### 1. Probe 2 still wins on val PPL after the QK-norm isolation

The round-1 caveat was that Probe 2 (Lorentz scores + QK-norm + warmup) bundled
QK-norm with the geometry, so its 11 PPL gap over the previously-published
Euclidean V3 baseline could be entirely from QK-norm. To isolate, this round
trained an Euclidean baseline with the same QK-norm and same training recipe
(LR=6e-4, 9000 steps, seed=42, WikiText-2, NANO d=128).

Stochastic 50-batch best-val PPL (same eval method as the training-time numbers):

| Variant                                                    | Best val PPL |
|------------------------------------------------------------|-------------:|
| Probe 2 (Lorentz scores + QK-norm + warmup, LR=6e-4)       |    **242.69** |
| **Euclid + QK-norm @ LR=6e-4 (NEW)**                       |    **253.62** |
| Probe 1 (hyper-fixed K=-1→-10 warmup, LR=6e-4)             |        269.28 |
| exp_b k10 (hyper-fixed K=-10, LR=6e-4)                     |        283.70 |
| Euclid V3 (LR=3e-4, vanilla, checkpoint lost)              |        277.10 |

**The geometry contributes ~11 PPL of improvement on top of QK-norm.** QK-norm
alone accounts for the rest of the gap (Euclid V3 → Euclid+QK-norm closes 23 PPL,
geometry on top adds another 11 PPL).

### 2. Probe 2 catastrophically degrades at long context

The same comparison under the full-split eval pack, NANO d=128 on WikiText-2:

| Variant                              |   id  |  ood  |  c256 |  c512 | c1024 | c2048 | c4096 | c4096 / c256 |
|--------------------------------------|------:|------:|------:|------:|------:|------:|------:|-------------:|
| Probe 2 (Lorentz scores + QK-norm)   | 254.9 | 380.3 | 254.9 | 281.0 | 367.4 | 476.9 | 616.8 |     **2.42×** |
| Euclid + QK-norm                     | 262.0 | 401.0 | 262.0 | 272.3 | 288.0 | 309.6 | 341.3 |         1.30× |

**Probe 2's PPL more than doubles from 256 → 4096 context, while Euclid+QK-norm's
grows by only 1.3×.** At 4096 tokens, Probe 2 is ~80% worse than Euclid+QK-norm,
having completely lost its training-context advantage.

For context, the V1-era hyper-fixed and hyper-perhead K=-1 models (no QK-norm,
no warmup, no curvature amplifier) have *excellent* long-context retention
despite their high absolute PPL:

| V1 variant (NANO d=128, step 4999)   |  c256 |  c512 | c1024 | c2048 | c4096 | c4096 / c256 |
|--------------------------------------|------:|------:|------:|------:|------:|-------------:|
| hyper-fixed K=-1 (LR=3e-4)           | 327.8 | 333.8 | 342.6 | 354.6 | 372.4 |    **1.14×** |
| hyper-perhead K=-1 (LR=3e-4)         | 326.5 | 333.5 | 344.1 | 358.3 | 377.5 |    **1.16×** |
| Euclid (LR=3e-4)                     | 280.0 | 290.8 | 313.2 | 341.3 | 374.8 |        1.34× |

So K=-1 hyperbolic *with* the |K| score multiplier (which is just ×1) retains
long context well. K=-10 hyperbolic with the same multiplier (×10) breaks
long context. The most plausible mechanism: at K=-10 the score formula
`|K|·(-⟨q,k⟩_L) / √d_head` produces score magnitudes ~10× larger than at K=-1,
making the softmax much sharper. At training context (256) the model learned
to operate in this regime; at 4× / 16× context, the sharpness over a larger
candidate set saturates onto the wrong tokens.

This is a clean explanatory story for why the hyperbolic "high curvature is
better" finding from Experiment B (K=-10 best in float32) does not translate
into long-context generalisation: high curvature buys in-distribution PPL at
the cost of length extrapolation. Worth confirming by running Probe 2 with
a milder K=-3 or K=-5 — if long-context degradation scales with |K|, the
mechanism is locked in.

---

## OOD picture is uniform across variants

OOD gap (`ood_ppl − id_ppl`) is similar across all WikiText-2 variants at
similar scale, ~120–180 PPL. The hierarchical-generalisation argument predicts
hyperbolic should have a *smaller* OOD gap; that is not visible.

(Probe 2 has a slightly smaller OOD gap than Euclid+QK-norm: +125.4 vs +139.0,
which is interesting but small.)

---

## rare_word_ppl now usable, but still uninformative

After the per-position loss fix, `rare_word_ppl` is finite and in the
2.5M–14M range across runs. The absolute numbers are huge because rare tokens
(bottom decile by frequency in train) are genuinely high-PPL for any small
model with vocab 50257. The interesting question is *relative* rare-word PPL
across variants — and the cross-variant ordering tracks `id_val_ppl` almost
exactly. No variant has a disproportionately *better* rare-word PPL than its
overall PPL would predict. The hierarchical-tail hypothesis remains
unsupported.

---

## Damage from this round

**Checkpoint overwritten:** I launched the Euclid+QK-norm training without
overriding `checkpoint_dir`, so `results/checkpoints/euclid/ckpt_step8999.pt`
was overwritten — losing `exp_d_euclid` (CodeParrot, val PPL 26.2). The
numeric result is preserved in `FINDINGS.md`, but the checkpoint is gone.

**Lost evaluation:** Last round I flagged "long-context crossover on
CodeParrot at ctx=1024" (exp_d_euclid 40.5 vs hyper-perhead 43.1) as the
strongest "wins where val PPL doesn't show it" signal in the whole programme.
I cannot extend that to 2048/4096 because exp_d_euclid is gone. The
hyper-perhead-on-code checkpoint is intact:

| exp_d_perhead_k10 (NANO d=128, CodeParrot)   |  c256 |  c512 | c1024 | c2048 | c4096 |
|----------------------------------------------|------:|------:|------:|------:|------:|
|                                              |  34.9 |  37.6 |  43.1 |  50.9 |  58.4 |

That's a 1.68× degradation 256→4096, between Probe 2's 2.42× and Euclid's
1.30×. Without the Euclidean code baseline I can't say whether the crossover
holds. Restoring it requires a fresh `exp_d_euclid` rerun (~80 min).

---

## What this changes about the headline

Going into this round, the programme's revised headline was *"Probe 2 (scores-only
+ QK-norm + warmup) beats Euclidean V3 on WikiText-2 val PPL, but the QK-norm
contribution is unisolated."* That caveat is now closed: **the geometry
contributes ~11 PPL on top of QK-norm at training context length** on
WikiText-2 NANO.

The new caveat is **length generalisation**: that 11 PPL win exists at the
trained context (256 tokens) and OOD text of the trained length. At 4× /
16× training length, Probe 2 falls catastrophically behind Euclid+QK-norm.
For any application where the model sees longer-than-trained sequences,
Probe 2 is strictly worse.

This pair of findings (geometry wins short, loses long) is more interesting
than "geometry loses everywhere." It is also a cleaner story than the
CodeParrot long-context crossover I lost: the same K=-10 + QK-norm recipe
that wins at training length is what loses at long length, so the trade-off
is intrinsic to the recipe and not data-specific.

---

## Next decisions

Worth doing:

1. **Probe 2 with K=-3 or K=-5** instead of K=-10 — if long-context degradation
   scales with |K|, the |K|-amplified score mechanism is confirmed. ~80 min.
2. **Rerun exp_d_euclid** to recover the checkpoint and test the CodeParrot
   long-context crossover at 2048/4096. ~45 min.
3. **`results/checkpoints/` cleanup** — change `train.py:save_checkpoint` to
   use `run_id`-keyed paths so future runs don't silently overwrite each other.
   ~30 min plus a re-test.

Not worth doing (yet):

4. Probe 2 on CodeParrot. Hold off until we know whether the long-context
   degradation is intrinsic to Probe 2 (point 1 above).

---

## Files written/modified

- `src/eval/ood_eval.py` — stride fix (round 1), rare-word per-position fix
  (round 2), multipliers (1, 2, 4, 8, 16)
- `src/attention.py` — `qk_norm` flag added to `EuclideanAttention`
- `src/model.py` — wires `qk_norm` config through to attention
- `scripts/train.py` — propagates `qk_norm` config to model
- `scripts/run_ood_eval_all.py` — propagates `qk_norm` to constructed model
  during eval-time loading
- `configs/euclid_qknorm_wikitext2.yaml` — NEW
- `results/eval/AGGREGATE.json` and per-checkpoint `eval_summary.json` files
- `results/logs/euclid_qknorm/euclid_qknorm_wikitext2_{log.json,train.jsonl}` — NEW
- `results/checkpoints/euclid/ckpt_step8999.pt` — overwritten (was
  `exp_d_euclid`, now `euclid_qknorm_wikitext2`)
