# Validation plan for the revision (pre-writing phase)

Purpose: before touching `paper.tex`, figure out **which reviewer claims are correct**, **which
of our current claims survive scrutiny**, and **what new analysis/code is actually needed** to
answer the reviewers with evidence rather than argument. This is not the revision/response plan
itself — it's the work that has to happen first so that plan can be written honestly.

Context gathered so far:
- `study/evaluation.py` + `study/config.py`: model selection uses `n_repeats_ms=20, n_splits_ms=5`
  CV per dataset to pick `(max_depth, min_samples_leaf)` maximizing AUC/r2 **with the default (`\le`)
  operator only**; the same dataset is then re-evaluated with `n_repeats=400, n_splits=5` CV using
  those selected hyperparameters. This is a non-nested procedure — hyperparameters are chosen on
  data that overlaps with every subsequent test fold.
- `study/007-analysis-presence.ipynb` and `008-analysis-proposed-method.ipynb`: `scipy.stats.wilcoxon`
  is applied directly to the vectors of per-fold scores from the 400×5 = 2000 repeated-CV
  iterations, treated as 2000 paired observations. No fold-overlap correction, no repeated-CV
  variance correction (e.g. Nadeau & Bengio), no unpaired alternative is computed anywhere in the
  pipeline currently.
- No synthetic/simulated data generator exists in `conditioning_bias/` or `study/`.
- No multi-class classification path exists (`_classifiers.py` is binary AUC-oriented).
- `README.md` describes an 8-hour, 8-notebook manual pipeline with a Google Drive link for
  intermediate `.csv`s — matches reviewer 2's "code drop" complaint.

This means **reviewers 2 and 4 are pointing at a real, verifiable methodological gap** (points 4
and 5 of reviewer 4, matched by reviewer 2's "paired test" concern and reviewer 1's "any tiny
effect can be made significant with n=400" observation). This is the highest-priority thing to
resolve, because it's the load-bearing criticism: if the significance tests are invalid, most of
the paper's central claims (Tables 2/3, the abstract's "statistically significant" language) are
unsupported as currently computed, regardless of how the surrounding text reads.

---

## 1. Critical: re-examine the statistical testing methodology (R1, R2, R4-4, R4-5)

This is the one item that can sink the paper if not resolved, and the one every other reviewer
either explicitly or implicitly flags. Needs to be settled **before** deciding whether new full
experiment runs (8h each) are worth doing.

### 1a. Non-independence of repeated-CV folds inflating the Wilcoxon test (R2, R4-5, R1-19)
- **Reviewer's claim**: treating all 2000 repeated-CV scores per dataset as independent paired
  observations is invalid because training/test sets overlap heavily across repeats; apparent
  sample size can be inflated arbitrarily by increasing repeats, manufacturing significance for
  arbitrarily small effects.
- **Validate**: this is textbook-correct (Dietterich 1998; Bengio & Grandvalet 2003, the exact
  reference reviewer 4 cites). There isn't really a defensible counter-argument here — the current
  test is *known* to be anti-conservative under repeated K-fold CV.
- **What to actually check on existing data** (no new experiments needed, re-analysis of existing
  `.csv` results in `study/data`):
  - Re-run reviewer 2's own claim: pull the per-fold scores, aggregate to one mean score per
    repeat (400 independent-ish repeat means instead of 2000 fold scores) or per outer split, and
    re-test. See how many of the currently-significant cells in Tables 2/3 survive.
  - Compute an **unpaired** test as reviewer 2 suggests, and compare which conclusions change.
  - Try a corrected variance estimator for repeated K-fold CV (Nadeau & Bengio corrected
    resampled t-test, or Bouckaert & Frank's corrected paired t-test) as a principled replacement
    for both raw-Wilcoxon and naive-unpaired approaches — this directly answers reviewer 4's
    citation of Bengio & Grandvalet (2003) that *no unbiased variance estimator exists for K-fold
    CV*, since the corrected estimators exist precisely to work around that.
  - Decide on a single defensible testing procedure, document why, and note its known limitations
    honestly (reviewer 4 will likely review the revision too).
- **Feasibility**: analysis-only, uses already-computed `study/data/evaluation_*.csv` — cheap,
  should be done first, no re-running of the 8-hour evaluation notebook required.

### 1b. Non-nested cross-validation / hyperparameter selection leakage (R3-11, R4-4)
- **Reviewer's claim**: hyperparameters are selected via CV maximizing default-operator
  performance, then the *same* selected hyperparameters are evaluated again via CV on overlapping
  data → optimistic bias for the default operator specifically, which could mechanically explain
  why `r_d^2` values skew negative in Table 2 (i.e., an artifact, not evidence about conditioning
  operators).
- **Validate**:
  - Confirm the mechanism: does optimizing hyperparameters *for the default operator* on the full
    data before re-evaluating create a structural advantage for the default operator in the later
    comparison? Reason through it concretely — model selection picks `(depth, leaf size)` that's
    best for `\le`; that hyperparameter choice is then reused for `<` scoring, which was never
    itself optimized. This is asymmetric by construction and is worth checking analytically before
    assuming nested CV is required.
  - Check whether Table 2's `r_d^2` sign asymmetry (reviewer 4's point 4.2, matching reviewer 1's
    point about "roughly as many negative as positive changes") lines up with this hypothesis, or
    is better explained by something else (e.g., dataset-specific noise).
  - Quantify how sensitive the reported hyperparameters are — for a sample of datasets, does
    re-running model selection with the `<` operator instead of `\le` pick meaningfully different
    `(depth, leaf size)`? If hyperparameters are nearly identical either way, the leakage concern
    is theoretical but practically small; if they diverge, it's a real confound.
- **Options to fix, in increasing cost**:
  1. Fix hyperparameters to sensible defaults (reviewer 4's suggested cheapest fix) and drop tuning
     entirely — cheapest, defensible, loses the "realistic evaluation" framing.
  2. Nested CV (inner loop tunes, outer loop estimates) — the textbook-correct fix, but 8h × (inner
     folds) compute cost, needs to be budgeted.
  3. Report both: current (biased) numbers *and* nested-CV numbers on a subset of datasets, to
     bound the size of the leakage empirically rather than committing to full nested CV everywhere.
- **Feasibility**: needs new compute; recommend prototyping on a handful of datasets before
  committing to a full re-run across all 40 datasets × 4 estimator variants.

### 1c. "Tiny effect, huge n" concern generally (R1-19, R1 overall recommendation)
- This is really the same root cause as 1a viewed from a different angle (reviewer 1 doesn't cite
  Bengio & Grandvalet but makes the identical observation). Resolving 1a substantially answers this.
- Separately worth validating: is the *practical* magnitude of the effect (0.02–1.5 percentage
  points, per the abstract) actually being oversold by the framing/language, independent of the
  significance-testing bug? This is a writing/tone issue (see section 5) but should be judged only
  after 1a is resolved, since the corrected effect sizes might differ from what's in the current
  tables.

---

## 2. Requested new experiments

### 2a. Synthetic / simulated data (R1-13, R2-6)
- **Why they want it**: unlimited test-set size removes finite-sample noise from the
  presence/absence of the bias and from the significance testing, and avoids "speculation on how
  trees would fare on the full data" (directly relevant to sections 1a/1b above too — synthetic
  data with a known ground truth lets you check nested-vs-non-nested CV bias directly, since you
  can generate a near-infinite validation set).
- **Design needed**:
  - Generate data with a controllable equidistant/discrete feature (e.g., integer feature with
    known small domain) alongside continuous features, with a known target function, so the
    "hit rate" of biasing nodes and the resulting prediction discrepancy can be computed exactly
    against a huge held-out set instead of estimated via CV.
  - Vary: domain granularity of the discrete feature, tree depth/regularization, sample size, and
    signal strength, to produce a controlled sensitivity study — this can also directly answer
    reviewer 1's request in 1a (does the bias emerge more/less with different regularization
    levels) and reviewer 4's implicit question about whether CV-based operator selection tracks
    full-data performance (can be checked exactly here, since "full data" performance can be
    approximated with a near-infinite test set).
  - This also gives a clean, low-cost way to directly test the CV-tuning-of-the-operator question
    from 3a below, since ground truth is available.
- **Feasibility**: new code (`conditioning_bias` has no synthetic generator yet), but computationally
  cheap compared to the real-dataset pipeline. Good candidate for early prototyping since it can
  also validate/refute several other reviewer claims cheaply (1b, 3a) before committing to
  expensive real-dataset re-runs.

### 2b. Cross-validation-based selection of the conditioning operator as a comparison baseline (R4-1)
- **Reviewer's claim**: the paper dismisses CV-based tuning of the operator as unviable, but the
  reviewer's detailed reasoning suggests it should work about as well as the proposed
  averaging/mirroring method in the worst case, and possibly better in the best case. The reviewer
  is explicitly asking the authors to **run this as an additional compared method**, not just argue
  about it in prose.
- **Validate**: this is directly testable using the *existing* evaluation infrastructure — add a
  third "mode" alongside default/non-default/averaged: pick whichever operator scored better in
  (inner) CV and use it. Compare its performance against the averaging method used in Table 3.
  - Concretely: does "pick-the-CV-winner" ever perform worse than "average-both" with statistical
    significance, and vice versa? The current argument in §3.1 (`sec:hyper`) is prose-only; the
    reviewer is right that it can and should be an empirical comparison instead.
  - This interacts with 1b: doing this properly means the operator-selection step must itself be
    nested to avoid the same leakage problem being used to argue against it.
- **Feasibility**: moderate — needs a new estimator "mode" (already have the `set_mode` abstraction
  in `_classifiers.py`/`_regressors.py`) and reuses the existing evaluation loop; real cost is
  compute time for another 400×5 CV pass per dataset/estimator, likely combinable with the 1b
  nested-CV re-run so both are addressed together.

### 2c. Multi-class classification (R5-2)
- **Reviewer's ask**: check whether conditioning bias behaves differently outside binary
  classification/regression.
- **Assess scope first**: is this a "must fix to get accepted" ask or a "future work" ask? Reading
  reviewer 5's review as a whole (fairly high-level, less technically pointed than 2/4), this reads
  more like a suggestion for broader framing than a hard requirement. Recommend validating this by
  scoping the effort (how much of `_classifiers.py`/`evaluation.py` is binary-AUC-specific) before
  committing, and defaulting to addressing it as a documented limitation / future work item rather
  than a new experiment track, unless the editor's decision letter treats it as a required item.
- **Feasibility if pursued**: `_classifiers.py` and the AUC-based evaluation would need real
  changes (multi-class AUC via OvR/OvO, macro-averaging) — nontrivial, not a small add-on.

### 2d. Impact analysis restricted to instances that actually hit a biased node (R3-13)
- **Reviewer's ask**: (i) report performance computed *only* on the subset of test instances that
  actually traverse a biasing node/threshold, and (ii) report what fraction of instances hit such
  nodes per dataset.
- **Validate/build**: the repo already computes `\rho`/`\rho_k` (proportion of *nodes* with
  thresholds on domain values, in `study/004-lattice-split-counts.ipynb`) but not the proportion of
  *test instances* actually routed through such a node, nor performance conditioned on that subset.
  This is a natural, cheap addition: instrument prediction to flag when `x_f` exactly equals a
  node's threshold along the traversed path, then recompute AUC/r2 restricted to the flagged
  subset. This should make the effect size much larger and easier to interpret (directly
  addresses reviewer 1's "effect is miniscule" complaint too, since the diluted, dataset-wide
  average is exactly what makes the numbers look tiny).
- **Feasibility**: straightforward instrumentation on top of `conditioning_bias/_tree_inference.py`
  and the existing evaluation pipeline; can likely be run on already-fitted models without a full
  8-hour re-run, since it only needs to change how predictions are scored, not how CV is done —
  but should be sequenced after 1a/1b since the underlying evaluation/testing procedure may change
  anyway.

### 2e. Explanation for why bootstrap aggregation doesn't average out the conditioning bias in RF (R1-12)
- Not an experiment so much as an analytical/derivation task — the reviewer wants a clear
  explanation, not just an empirical observation, of why bagging many trees doesn't wash out a
  systematic bias that's shared by (nearly) all trees in the forest (since every tree in the forest
  sees a similarly-distributed feature and similar threshold-alignment issue, bagging averages
  *variance* across trees but the conditioning-operator choice is a *constant* applied identically
  to every tree, so it isn't a source of cross-tree variance that bagging would reduce).
- **Validate**: this reasoning should be checkable against the existing presence-table data
  (`\rho` for RF vs DT) — if it's right, RF nodes should show *similar or higher* proportions of
  biased nodes than DT nodes (per the paper's own §2.2 regularization argument), and the
  bias-affected-instance analysis from 2d, run per-tree within a forest, should show correlated
  (not independent) bias direction across trees rather than averaging to zero.
- **Feasibility**: reanalysis of existing data plus the 2d instrumentation; no new full experiment
  run required.

---

## 3. Items that are re-analysis or rebuttal, not new experiments

### 3a. Table 2/3 reporting: relative (%) figures alongside absolute deltas (R1-16)
- Pure reporting change: existing `.csv` results already contain the absolute mean scores needed
  to compute relative deltas; just extend `study/008-analysis-proposed-method.ipynb` /
  `007-analysis-presence.ipynb` to also compute `auc_d / auc_leq` (or similar) and add columns.
  Validate that reviewer 1's suspicion is correct first — i.e., check whether the "≈1%" relative
  figures quoted in the current prose text are actually consistent with the absolute deltas in
  Table 2, since reviewer 1 explicitly says they don't currently add up ("hard to believe... of the
  order of one percent"). This needs to be checked/reconciled, not just reformatted.

### 3b. Absolute AUC/r2 values for both operators in Tables 2/3, not just differences (R4-6)
- Also a reporting change on existing data — the underlying evaluation already produces per-operator
  mean scores, they're just not currently surfaced in the LaTeX tables.

### 3c. Sensitivity to K (larger K reduces train/full-data mismatch) (R4-2)
- Reviewer 4 suggests K=10 or K=20 instead of K=5 would make within-CV models closer to full-data
  models. Worth a targeted sensitivity check on a handful of datasets (not the full 40) comparing
  `\rho_k` at K=5 vs K=10/20 against `\rho` (full-data), to see whether this changes the picture
  materially. Cheap relative to a full re-run, and gives a concrete empirical answer instead of
  prose back-and-forth.

### 3d. Reviewer 4's conceptual arguments (points 1, 3, 20, 21, 26) about whether CV-tuning is really
inferior to averaging, and whether "expected to be at least as good as the worse operator" is a
weak/misleading claim
- These are logical/argumentative points about what "expected performance" of averaging vs.
  tuning vs. random-pick actually means. Recommend working through the expected-value algebra
  explicitly (what is E[score] for: pick-default, pick-cv-winner, average-both, oracle-best) so the
  revision can either concede reviewer 4 is right (averaging's expected performance under "50/50 a
  priori" framing is genuinely identical to random-pick, and CV-tuning should be expected to beat
  random-pick whenever the operator's effect correlates at all between CV-fold-trained and
  full-data-trained trees) or provide a precise counter-argument with evidence from 2b/2a. This
  should fall out naturally once 2a/2b are done — treat as synthesis work, not new data collection.

### 3e. Reviewer 3's specific technical/example checks (R3-4, R3-6, R3-9, R3-10, R3-12)
- CART categorical/oblique-split claim (R3-4): factual correction, verify against Breiman et al.
  1984 and current sklearn docs (sklearn's CART implementation does *not* support categorical
  grouping or oblique splits directly — worth confirming precisely what to correct, since the
  reviewer's claim is about CART-the-algorithm, not sklearn-the-implementation, so the wording in
  the paper likely needs a scoping fix rather than a factual reversal).
- Classification aggregation-tie clarification (R3-6): check what `_classifiers.py` actually does
  when the two conditioning predictions disagree on class — is it averaging probabilities (which
  sidesteps ties entirely) or hard-voting? Read the code before writing the clarifying text.
- Worked example on page 9 and Figure 3 label (R3-9, R3-10): verify against the actual sample data
  in the appendix table and the actual `forest*.pdf`/`discrete.pdf` figures — these might be simple
  factual errors, check by hand against Table `tab:sample`.
- Table 1 alpha/beta inconsistency (R3-12): check `study/002-dataset-summary.ipynb` / the generated
  `tab_datasets_*.tex` to see whether this is a rendering bug (e.g., blank cell should show N/A) or
  a genuine inconsistency in what's reported.

---

## 4. Reproducibility complaint (R2)

- **Claim**: "typical code drop... not a bunch of half-executed Jupyter notebooks."
- **Validate**: check whether the notebooks in `study/` are currently saved with stale/empty
  outputs, whether they run end-to-end without manual intervention, and whether the README's
  "8 hours" pipeline can realistically be reproduced by a reviewer without hand-holding.
- Concrete option to evaluate (not necessarily commit to): whether converting the 8-notebook
  manual pipeline into a small number of scripts (or a single orchestrating script/Makefile that
  calls notebook-derived functions) is worth doing given the rest of the revision workload — this
  is real engineering effort, should be weighed against the statistical fixes above, which are
  higher priority for getting the paper accepted at all.

---

## 5. Writing/framing items (deferred, no validation needed — just noted for the actual revision)

Shortening the paper, reducing citation/review-paper feel, toning down language, terminology fixes
("prediction" not "inference", "variation" vs "bias", removing "integrating out", \citep vs \citet,
CART citation year, etc.), and the many small wording/typo items from reviewer 4's "minor essential
revisions" and reviewer 3's typo list. These don't require validation or experiments — they're
direct edits for the revision-writing phase, deliberately excluded from this plan.

---

## Suggested sequencing

1. **Re-analyze existing CV results with corrected/alternative significance tests** (§1a) — cheap,
   fast, and determines how much of the rest of the paper's numeric story survives before
   investing in anything else.
2. **Reconcile the relative-vs-absolute numbers in Table 2** (§3a) and **check the hyperparameter
   asymmetry hypothesis** (§1b, analytical part) — cheap, clarifies whether the negative-skew in
   `r_d^2` is a real finding or a leakage artifact.
3. **Build the synthetic-data generator** (§2a) — moderate effort, but it's the single piece of new
   infrastructure that lets you cheaply validate multiple reviewer claims (1b, 3a/3d, presence of
   bias, RF-doesn't-average-out) against exact ground truth before committing to expensive re-runs
   on the 40 real datasets.
4. **Decide on nested-CV / hyperparameter-fixing strategy** (§1b options) using evidence from 1–3,
   then budget and run the real-dataset re-evaluation, folding in the CV-operator-selection
   baseline (§2b) and the biased-instance-subset analysis (§2d) into the same re-run so the
   expensive 400×5×40-dataset computation is only paid for once.
5. **Multi-class scoping decision** (§2c) and **reproducibility engineering decision** (§4) — lower
   priority, revisit after 1–4 clarify how much budget/time is left.
6. Only after 1–5 land on solid ground: write the actual point-by-point response and revise
   `paper.tex`/tables using the validated (possibly revised) numbers.

## Open questions for you before starting execution

- How much compute/time budget is realistic for re-running the 8-hour evaluation pipeline
  (possibly multiple times: nested CV, operator-CV-selection baseline, larger-K sensitivity)?
- Is Machine Learning (Springer) still the target venue given reviewer 1's outright reject
  recommendation, or is this revision aimed at a resubmission strategy that also considers venue?
- For §2c (multi-class): in scope for this revision, or explicitly deferred to "future work" in the
  text?
- For §4 (reproducibility): worth the engineering time now, or acceptable to address minimally
  (e.g., fix notebook outputs, clarify README) given everything else on the list?
