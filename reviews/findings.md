# Validation findings (live document)

Tracks confirmed results from executing `reviews/validation_plan.md`. Written as findings are
confirmed, not after the fact — this is the evidence base the actual revision/response will draw
from. Scripts referenced here currently live in the session scratchpad; the ones worth keeping
long-term should be moved into `study/` once the analysis approach is finalized (not yet done).

Status: §1a done (decisive). §3a done. §2e done. §3e mostly done. §1b (hyperparameter asymmetry)
and the larger-K real re-run are in progress. Synthetic data (§2a) and the CV-operator-selection
baseline (§2b) not started. Multi-class (§2c) and reproducibility engineering (§4) explicitly
deferred per user decision.

---

## §1a. Significance testing re-analysis — CONFIRMED, decisive

Re-analyzed the existing `study/data/evaluation_{dtc,dtr,rfc,rfr}.csv` (already computed, no new
experiments) with four testing methods, on all 240 (estimator × dataset × mode-pair) cells:

| Method | % significant (α=0.05) |
|---|---|
| Current paper method (paired Wilcoxon on 2000 pooled fold-scores) | 54.6% (131/240) |
| Repeat-level aggregation (Wilcoxon on 400 per-repeat means) | 59.6% (143/240) |
| **Corrected resampled t-test** (Bouckaert & Frank 2004, correcting for k-fold train/test overlap) | **1.2% (3/240)** |
| Unpaired test (reviewer 2's suggestion) | 0.8% (2/240) |

Of the 131 cells the paper currently calls significant, only 3 survive the corrected test — all
from `plastic` (RF regression, `<=` vs `<` and both vs `avg_half`). Everything else, including
cells with naive p-values around 1e-200, is statistically indistinguishable from noise once the
repeated-CV non-independence is properly corrected for.

**Conclusion**: reviewers 1, 2, and 4 are correct — the paper's central significance claims
(Tables 2/3, and the abstract's "statistically significant improvements") are not supportable as
currently computed. This is the load-bearing finding of the whole revision.

Distribution of corrected p-values across all 240 cells: 3 < 0.05, 5 < 0.1, 6 < 0.2, 6 < 0.3,
12 < 0.5. I.e. it's essentially a step function — `plastic` shows a large, robust effect; almost
everything else lands at p > 0.4, not "almost significant."

## §1a/§3c. Why more repeats won't help, and why larger K should (math + pilot)

The corrected variance is `var_d * (1/n + 1/(k-1))`, n = n_repeats × k. As n_repeats → ∞, the
`1/n` term → 0, so the corrected variance approaches a **floor of `var_d/(k-1)`** that does NOT
shrink with more repeats — only increasing k shrinks it. Verified empirically by subsampling the
existing 400 repeats down to 25: even at n_repeats=25, the `1/(k-1)` floor already dominates the
`1/n` term by >30x, and going to 400 repeats only pushes that to ~500x. Confirmed on 5 borderline
cases (`plastic` dt/rf, `bupa` rfc, `wankara` rfr, `excitation_current` rfr): corrected p barely
moves between n_repeats=25 and 400.

Formula-based projection (holding `var_d` fixed at its observed k=5 value — an approximation) for
larger k on these same 5 cases suggested k=10–30 would push several of them under 0.05. **A real
pilot re-run (not just the formula) is in progress** (`pilot_larger_k.py`, background job): refits
the actual estimator at k=5/10/20/30 with n_repeats=100 (already enough to saturate the floor, per
the point above) on the same 5 borderline datasets, using the *existing* selected hyperparameters
(isolates the k effect from the hyperparameter-selection question in §1b). Results pending.

**Answer to the user's question** ("more repeats maybe?"): no — mathematically futile past roughly
n_repeats=50–100 for this test. The lever is k (fold count), not repeat count. This directly
operationalizes reviewer 4's own suggestion (point 2 of the major revisions).

## §3a. Relative vs. absolute reporting (reviewer 1, point 16) — checked, minor

Cross-checked every specific "N percentage points" example cited in the text against the raw
`evaluation_*.csv` means:

| Text claim | Dataset | Computed | Match |
|---|---|---|---|
| 0.3pp AUC | cleveland-0_vs_4 (dtc) | 0.31pp | yes |
| 2pp r² (presence) | o-ring (rfr) | 2.42pp | yes (rounded) |
| 0.02pp AUC (results) | haberman (dtc) | 0.022pp | yes |
| 0.1pp AUC (results) | bupa (rfc) | 0.121pp | yes (rounded down) |
| 0.5pp r² (results) | cpu_performance (dtr) | 0.560pp | yes (rounded down) |
| 1.5pp r² (results) | o-ring (rfr) | 1.549pp | yes |

All text mentions are consistently "percentage points" (an absolute measure), not relative
percentages — reviewer 1's literal claim of a text/table mismatch doesn't hold up on inspection.
The underlying ask (add relative-% columns for reader interpretability) is still worth doing when
the tables are rebuilt with corrected statistics, since it's cheap and genuinely helps readers
who don't intuitively know what a 1.5pp r² change means relative to typical r² magnitudes for that
dataset. Not a correctness issue, just a presentation enhancement — low priority, bundle into the
table rebuild once §1b/§3c settle on a final testing methodology.

## §2e. Why bagging doesn't average out the conditioning bias (reviewer 1, point 12) — evidence found

Compared ρ (proportion of nodes with thresholds falling on feature domain values, full-data fit)
between decision trees and random forests using the already-computed `study/data/splits_*.csv`:

| | DT mean ρ | RF mean ρ |
|---|---|---|
| Classification | 0.0184 | **0.0823** (4.5x higher) |
| Regression | 0.0589 | **0.1015** (1.7x higher) |

RF nodes are substantially *more* prone to threshold-domain-value alignment than DT nodes, not
less — consistent with the paper's own §2.2 argument that RF trees need less depth-regularization
and so are grown less pruned. This gives a two-part, evidence-backed answer to reviewer 1:
(a) individual RF trees are structurally *more* susceptible to the bias (higher ρ), so there's more
raw bias to average, not less; and (b) more fundamentally, the conditioning operator is a constant,
deterministic property applied identically by the same implementation to every tree in the forest —
it is not an independent, zero-mean source of noise across ensemble members that bagging would be
expected to cancel. Bagging averages out *variance across trees*; it does not cancel a *systematic,
shared* modeling choice. This reasoning + the ρ comparison above is enough to answer reviewer 1's
point directly in the revision without needing new experiments.

## §3e. Reviewer 3's specific factual/example checks — resolved

- **R3-9 (worked example inconsistency, page 9)**: CONFIRMED as a real error. Reproduced the exact
  sample-data tree (sklearn 1.5.1, `random_state=5`) — root split is `self-employed <= 0.5`. The
  example vector `x = (self-employed: 0, dependents: 3, income: 50k)` has `self-employed=0`, which
  routes LEFT at the root into the `income <= 70k` subtree — it never reaches the `dependents <= 3`
  node at all (that node only exists in the `self-employed > 0.5` branch). Reviewer 3 is right: the
  example needs `self-employed: 1` to actually reach the node being discussed. Fix: change the
  example vector's self-employed value (or pick a different illustrative vector).
- **R3-10 (Figure 3 typo, "-60k instead of -70k")**: CONFIRMED and precisely diagnosed. Read
  `forest.pdf`/`forest2.pdf`/`forest3.pdf` directly. Manually re-derived the correct mirrored tree
  from `forest3.pdf` (recursively swap left/right children and negate thresholds at every node, per
  the paper's own definition of mirroring in §3.3). The correct mirroring places `-income <= -70k`
  as the root's direct right child and `-income <= -60k` nested under `-dependents <= -3`; the
  actual `forest2.pdf` has these two thresholds **swapped** between those two positions. Confirmed
  `forest2.pdf`/`forest3.pdf` are hand-made (not notebook-generated — only `forest.pdf` has
  generating code in `study/000-illustration-decision-tree.ipynb`), so this needs a manual fix to
  the figure source, not a rerun of any notebook.
- **R3-6 (classification aggregation on disagreement)**: resolved by reading
  `conditioning_bias/_classifiers.py::predict_proba_average` — both DT and RF averaging modes
  average predicted **class probabilities** (`values_le`, `values_leq` normalized then meaned),
  never hard labels. There is no tie-breaking logic because none is needed — this should be stated
  explicitly in the text (currently ambiguous), which directly answers the reviewer's question.
- **R3-12 (Table 1 shows only α or only β per dataset)**: not a bug. Confirmed via
  `study/001-optimal-parameters.ipynb`: model selection sweeps `max_depth` and `min_samples_leaf`
  as two **separate 1-D sweeps** (not a joint 2-D grid) and keeps only the single best point across
  both sweeps — so exactly one of α/β is ever "the tuned one" per dataset, the other stays at its
  sklearn default. This is a genuine (if minor) methodology limitation worth flagging: hyperparameters
  are optimized one-at-a-time, not jointly. Needs a caption/text clarification either way.
- **R3-4 (CART categorical/oblique-split claim)**: the paper's "\emph{CART...assume[s] all features
  being continuous}" is inaccurate as a claim about CART-the-algorithm (Breiman et al. 1984 CART
  supports categorical grouping and has an oblique-split variant); it's only true of the specific
  *implementations* under study (sklearn, `tree`, etc.). Needs rescoping in the text, not a factual
  reversal — no further validation needed, this is a direct text fix.

---

## §1a/§3c. Larger-K pilot — real re-fits, not just formula projection — DONE

`pilot_larger_k.py`: real re-fit at k=5/10/20/30, n_repeats=100 (per the floor argument above,
already saturated), on the 5 borderline datasets, reusing the existing selected hyperparameters
(isolates the k effect). Result: **larger K is a real, selective tool, not a blanket fix**:

| dataset/estimator | k=5 | k=10 | k=20 | k=30 |
|---|---|---|---|---|
| `rfr / plastic` | sig (p=1e-4) | sig | sig | sig |
| `dtr / plastic` | not sig (p=0.086) | **sig (p=0.041)** | sig (p=0.014) | sig (p=0.013) |
| `rfc / bupa` | not sig (p=0.56) | not sig (0.49) | not sig (0.57) | not sig (0.53) |
| `rfr / wankara` | not sig (p=0.45) | not sig (0.30) | not sig (0.27) | not sig (0.24) |
| `rfr / excitation_current` | not sig (p=0.46) | not sig (0.38) | not sig (0.38) | not sig (0.35) |

2 of 5 near-misses genuinely cross into significance as k grows (and `dtr/plastic`'s mean_diff
itself grows with k: 1.2e-3 → 2.2e-3, i.e. not just a variance artifact — the estimated effect
size itself gets larger and more stable). 3 of 5 stay flat regardless of k — larger k is not
manufacturing significance everywhere, which is reassuring: it's tracking real vs. noise signal,
not inflating power indiscriminately.

## §2a. Synthetic data generator + significance sweep — DONE, this is the key structural finding

Built `synthetic.py`: a generator with a controllable discrete/equidistant feature (domain size,
step) alongside continuous filler features, known generative process for both regression and
classification. Validated it reproduces realistic lattice-aligned-threshold rates (2-5%,
comparable to the real datasets' measured 2-10%) — see stress test in scratch notes, not
duplicated here.

Built `synthetic_significance_demo.py`: per replicate, draw a **fresh** training set, fit once,
then draw a **fresh, large (200k), independent** i.i.d. test set to score every mode. Repeat for
200 fully independent replicates. This sidesteps the entire repeated-CV correlation problem by
construction — no correction needed, an ordinary paired test is valid.

Full sweep (DecisionTreeRegressor and RandomForestRegressor, domain sizes 5/11/21/51, depths
3/5/8/12, `n_train=500`, `n_test_eval=200000`, `n_replicates=200`):

- **`<=` vs `<` (which individual operator is better): never significant, in any of the 11
  configurations tested** (Wilcoxon p ranges 0.07–1.0, always ≥0.05; typically ≥0.3). Mean
  differences are ~0 in every case. Individual operator superiority does not appear to be a real,
  reproducible population-level property — consistent with reviewer 4's theoretical argument (his
  point 1) that *a priori* neither operator should be systematically better.
- **`avg` vs `worst(<=, <)` (the paper's proposed method vs. the worse individual operator): highly
  significant every time the tree has enough depth to form biased nodes at all** — p from 1e-4 down
  to 1e-34 across every domain size at depth 8, and both depth=8 and depth=12 in the depth sweep.
  At depth 3 and depth 5, the effect is exactly zero (std=0 at depth 3, std=0.00001 at depth 5) —
  trees this shallow essentially never create lattice-aligned thresholds, matching the paper's own
  regularization argument in §2.2/§3.4.

**This is the cleanest, most decisive result of the whole validation effort.** Two independent
lines of evidence — the corrected test on real data (§1a) and the fully-valid synthetic replicate
test (§2a) — agree: claiming a *specific* conditioning operator is reliably worse is not
supportable. But the paper's actual proposed fix (average both operators, beat the worse one) is a
robust, real, statistically valid effect, confirmed by a test design that owes nothing to the
repeated-CV methodology under dispute. This suggests the defensible core contribution for the
revision is centered on the averaging method's robustness/risk-reduction property, not on
"conditioning bias is a widespread, individually-detectable problem across real datasets."

## §1b. Hyperparameter-asymmetry sweep — DONE, real but narrowly confined to one dataset

`model_selection_lt.py`: reruns the *exact* model-selection procedure of
`study/001-model-selection-{md,msl}.ipynb` and `001-optimal-parameters.ipynb`, but scored with `<`
instead of the hardcoded `<=`, across all 40 datasets × 4 estimators. Tests whether tuning for the
non-default operator picks different hyperparameters than tuning for the default — the mechanism
reviewer 4 (point 4) and reviewer 3 (point 11) both flagged as a possible source of the
negative-skewed `r_d^2` values in Table 2.

(Note: the script's own printed `params_differ` column is a false-positive artifact — it compares
raw dict-repr strings, and `np.float64(0.2)` vs `0.2` print differently across the two numpy
environments used for the original vs. this run despite being the same number. All numbers below
use a corrected, value-based comparison instead.)

| estimator | # datasets w/ genuinely different hyperparameters | max \|score diff\| | median \|score diff\| |
|---|---|---|---|
| dtc (AUC) | 4 / 20 | 0.0021 (0.21pp) | ~0 |
| rfc (AUC) | 9 / 20 | 0.0017 (0.17pp) | ~0 |
| dtr (r²) | 6 / 20 | **0.2229 (22.3pp)** | 0.00013 |
| rfr (r²) | 3 / 20 | **0.0205 (2.05pp)** | 0.00019 |

**Every single large deviation, in both regression estimator groups, is the same one dataset:
`o-ring`.** Every other dataset × estimator combination (39/40 datasets, all of classification) is
negligible — well under 0.2pp, mostly indistinguishable from zero. So the hyperparameter-selection
leakage reviewers 3 and 4 flagged is real, but it is **not a pervasive confound across the study
— it's narrowly confined to one fragile dataset**, not a general property of the non-nested
procedure.

`o-ring` has only **23 records**. 5-fold CV model selection on 23 rows is inherently unstable:
tuning for `<=` picks `min_samples_leaf=0.07` (dtr, r²=0.127) vs. tuning for `<` picking
`max_depth=None` (r²=0.350) — a 22-point swing driven by which operator the small, noisy inner CV
happened to reward, not by any real property of the conditioning operators. For `rfr` (the actual
estimator/dataset combination the paper's abstract highlights — "1.5 percentage points... in the
most sensitive case of random forest regression"), both tunings land on the same `max_depth=2`,
but the achieved r² still differs by ~2.05pp — the same order of magnitude as the claimed headline
effect itself.

**Implication for the revision**: the paper's own flagship example (`o-ring`) is very likely
substantially confounded by hyperparameter-selection instability on an extremely small dataset,
independent of everything found in §1a/§2a. This needs to be addressed directly — options include
dropping `o-ring` from the pool (23 records is arguably too small for reliable 5-fold-CV-based
hyperparameter selection regardless of the conditioning-bias question), nested CV specifically for
it, or at minimum no longer citing it as the headline "most sensitive case" example without
qualification.

## §New. Real-data K-sweep + cross-dataset meta-analysis — DONE, this is the strongest evidence yet

Following up on the reframed headline ("operator choice is arbitrary, averaging is a robust
insurance policy"), built a proper real-data protocol to test it directly: replaced `o-ring` and
`diabetes` (both too small for stable 5-fold hyperparameter selection, confirmed by a dedicated
stability spot-check — `concrete`/`boom_bikes`, their replacements, pick *identical* hyperparameters
whether tuned for `<=` or `<`, with score differences ~0.0002-0.006pp vs. `o-ring`'s 22pp), then
ran a K ∈ {5,10,15,20} sweep (capped per-dataset so test folds stay ≥3 records; classification
self-caps at the minority-class count) across all 40 datasets × 4 estimators, using the corrected
resampled t-test throughout, reusing the existing K=5/400-repeat data for all continuing datasets
(only genuinely new compute: the 2 replacement datasets, and K=10/15/20 for everyone).

Added a **cross-dataset meta-analysis** on top: a sign/binomial test on win-rate across the
20-dataset pool (does `mean_diff`'s sign show a systematic pattern across datasets, independent of
any single dataset's own significance) plus Fisher's combined-p. This directly answers "is there
a real effect across the pool" even when no individual dataset has enough power to show it alone.

**`<=` vs `<` (individual operator comparison) — confirms the null finding, now on real data at
scale**: sign-test p ranges 0.31–1.0 across every estimator × K combination, essentially always a
~50/50 split of which operator wins per dataset, 0-1 individually significant datasets out of 20
in every case. This is the same conclusion as the corrected single-K test (§1a) and the synthetic
experiment (§2a), now confirmed a third way, at full scale, with a test purpose-built for exactly
this question.

**`avg` vs `worst(<=, <)` (the proposed method) — decisive, and gets stronger with K**:

| estimator | K | sign-test p | wins/losses (of 20) | Fisher combined-p | individually significant |
|---|---|---|---|---|---|
| dtc | 5 | 0.0005 | 12/0 | — | 0 |
| dtc | 20 | 0.016 | 7/0 | — | 0 |
| dtr | 5 | 0.00002 | 17/0 | 0.97 | 0 |
| dtr | 20 | 0.00003 | 16/0 | 0.67 | 1 |
| rfc | 5 | 0.000004 | 19/0 | 0.96 | 0 |
| rfc | 20 | 0.0002 | 13/0 | 0.65 | 2 |
| **rfr** | **5** | 0.00004 | 19/1 | 1.9e-4 | 2 |
| **rfr** | **10** | 0.000002 | 20/0 | **3.8e-9** | 8 |
| **rfr** | **15** | 0.00004 | 19/1 | **3.0e-11** | 10 |
| **rfr** | **20** | 0.0004 | 18/2 | **1.1e-12** | 11 |

The averaging method **never loses on the sign test** — across every estimator type and every K,
`avg` beats `worst(<=,<)` on more datasets than chance would predict, at high significance, and in
most (K,estimator) cells it *literally never loses on a single dataset* (0 losses). For random
forest regression specifically, the effect strengthens dramatically with K: individually-significant
datasets climb from 2/20 (K=5, the original setup) to 11/20 (K=20), and the Fisher combined p-value
reaches 1e-12.

**This is now the strongest, most complete evidence in the whole validation effort**, and it holds
up across three independent methodologies (corrected per-dataset test, synthetic ground truth,
real-data meta-analysis): claiming a *specific* conditioning operator is reliably worse doesn't
survive scrutiny, but the paper's actual proposed fix — average both operators — has a real, robust,
statistically overwhelming advantage over whichever operator turns out to be worse, and that
advantage is *detectable* at the whole-pool level even in the many individual datasets too small to
show it alone. This should be the new headline result.

(Caveat: for classification, K=10/15/20 aren't always achieved exactly — a handful of datasets have
minority-class counts below the requested K, so the self-cap lands on odd values like 6, 8, or 13
for 1 dataset each. Reported in the table above as its nearest standard K bucket for readability;
the raw per-K breakdown is in `study/data/ksweep_results.csv`/`meta_analysis_results.csv`.)

## Not yet started

- §2b CV-operator-selection as a real compared baseline (reviewer 4, point 1) — the synthetic
  generator now makes this cheap to add: fit on the training replicate, pick whichever operator
  scored better on a held-out slice of the SAME replicate, score on the fresh 200k test set. Natural
  next step once the hyperparameter-asymmetry job frees up compute.
- §2d biased-instance-subset analysis (reviewer 3, point 13)
- Decision on final testing methodology + full real-dataset table rebuild, informed by all of the
  above once the last job lands.
