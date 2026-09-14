**Status: v2 written, compiled cleanly (pdflatex + bibtex, 0 errors, 0 undefined refs/citations,
0 overfull hboxes), and spot-checked visually — see "Remaining polish" at the end.**

# Revision plan: mapping validated findings onto the paper

Companion to `reviews/findings.md` (the evidence) — this maps that evidence onto concrete changes
to `paper.tex`, section by section. The new revision lives at
`paper_2023_conditioning_bias_springer/paper_v2.tex` (original `paper.tex` untouched).

## Guiding principle

Keep the paper's narrative arc, structure, and citations intact. The core idea (conditioning bias
exists, mirroring/averaging fixes it, free for RF) is still correct and still the contribution.
What changes is: (a) the statistical methodology used to demonstrate it, (b) the headline framing
of *what* is robustly demonstrated, (c) the dataset pool (2 unstable datasets swapped), (d) a
handful of confirmed factual errors, (e) length/tone per reviewer 1's complaint.

## Section-by-section

**Abstract**: reframe the headline claim. Old: "statistically significant improvements of up to
0.1-0.2pp... 1.5pp in the most sensitive case." New: individual operator superiority is not a
reliably detectable population-level effect (confirmed three independent ways); the proposed
averaging method has a robust, statistically strong advantage over the worse operator, confirmed
via a corrected significance test, a synthetic ground-truth experiment, and a 40-dataset
meta-analysis whose evidence strengthens with fold count (Fisher combined-p down to 1e-12 for RF
regression at K=20). Keep the "free lunch for RF" framing — still true and still the practical
payoff.

**Introduction**: trim per reviewer 1 (moderately, not a rewrite), keep citation structure. Add
the CART 1984 book citation to the opening sentence (`cart` key already in refs.bib, already used
elsewhere in the paper — just add it here too). Rescope the "CART assumes all features continuous"
claim to the specific implementations studied, not CART-the-algorithm (reviewer 3). Update
contribution bullets 2 and 4 to match the corrected findings instead of the old effect-size
numbers. Add a short, explicit methodological contribution bullet: identifying and correcting a
significance-testing pitfall (naive Wilcoxon on pooled repeated-CV folds) that likely generalizes
beyond this paper.

**Problem formulation** (§2): mostly unchanged — well-received, not challenged on substance.
Terminology fixes: "inference" → "prediction" throughout (reviewer 4); define "equidistant domain"
on first use rather than later (reviewer 4); clarify "variation" vs "bias" wording (reviewer 4);
`\citep`→`\citet` where reviewer 4 flagged it; fix the worked example on the current page 9
(`self-employed: 0` → `1`, confirmed by reproducing the actual fitted tree — with `self-employed:
0` the record never reaches the `dependents` node being discussed at all); fix the swapped
`-70k`/`-60k` labels in the mirrored-tree figure (confirmed by re-deriving the correct mirroring
by hand — new figure being regenerated programmatically, see below).

**Proposed method** (§3): keep structure (operator-as-hyperparameter / integrating out / mirroring).
Strengthen §3.1's argument that CV can't select the operator with the new evidence: it's no longer
just theoretical, we now have direct empirical confirmation (synthetic ground truth AND the
40-dataset meta-analysis) that no operator is systematically better, so CV-based selection would
be tuning to noise. Add the §2.2/§2e finding on why bagging doesn't average out the bias: RF nodes
show 1.7-4.5x *higher* lattice-alignment rates than DT nodes (more regularization headroom, per
the paper's own existing argument), and more fundamentally the conditioning operator is a constant
applied identically by the same implementation to every tree — not an independent per-tree noise
source bagging would cancel. Add the explicit classification-aggregation clarification (probability
averaging, not hard voting — reviewer 3) and the "why not random tie-breaking" note (reviewer 4).

**Tests and results** (§4): the biggest rewrite, new evidence throughout.
- Dataset section: `o-ring`/`diabetes` replaced with `concrete`/`boom_bikes` (same
  smallest-eligible-by-size selection rule, next two in the ranking — not cherry-picked), following
  a documented, reproducible instability diagnostic (5-fold-CV hyperparameter selection on <50
  records is unstable regardless of conditioning operator).
- New subsection replacing the old naive-Wilcoxon methodology: the corrected resampled t-test
  (Bouckaert & Frank 2004) and why it's needed — this doubles as a direct, evidence-backed answer
  to reviewers 1, 2, and 4's shared statistical objection, turning their strongest criticism into
  an explicit contribution of the revision.
- Table 2 analogue (presence): rebuilt on corrected p-values.
- Table 3 analogue (proposed method): rebuilt on corrected p-values, `avg` vs `worst(<=,<)`.
- New table/subsection: the cross-dataset meta-analysis (sign test + Fisher combined-p across
  K=5/10/15/20) — this is the new central evidence for the paper's actual claim.
- New subsection: synthetic validation, summarizing the i.i.d.-replicate experiment (no CV
  non-independence issue by construction) as independent confirmation of the same pattern.

**Summary and conclusions** (§5): rewrite to match the new headline: bias exists and is
mechanistically real (confirmed via presence-table lattice-alignment rates, unchanged), individual
operator superiority is not reliably detectable (new, evidenced conclusion, a more honest and
defensible claim than before), the averaging method is a statistically robust, low/free-cost
improvement regardless (strengthened conclusion, better evidenced than the original submission).

## What's explicitly NOT changing

- The mirroring/tree-inference math and technique (§3.3) — untouched, was never challenged on
  correctness, only on a figure-label typo.
- The core dataset selection *methodology* (susceptibility criterion, UCI/KEEL sourcing) — only 2
  of 40 datasets swapped, same selection rule applied consistently.
- Citation style/bibliography structure — reuse `refs.bib` as-is; both new datasets' citation keys
  (`uci`, `boombikes`) already exist there.
- Anything not questioned by reviewers or contradicted by the validation work (e.g., the free-lunch
  RF argument, the tree-mirroring identity, the general problem motivation).

## Known gaps / not attempted in this revision

- Nested CV across the full 40-dataset pool (flagged as a stretch goal, not needed — the real
  problem it would have fixed, o-ring's instability, was resolved by dataset replacement instead;
  see the conversation record for the reasoning).
- CV-based operator selection as a literal compared baseline (reviewer 4 point 1) — addressed
  analytically + empirically via the "no operator is systematically better" finding instead, not
  via implementing operator-selection as a fourth method. Could be added later if reviewers still
  want it spelled out as a compared method rather than argued from the null-effect evidence.
- Reviewer 3's biased-instance-subset analysis — not run in this pass; noted as a natural
  follow-up, not blocking.
- Multi-class generalization (reviewer 5) — explicitly out of scope per earlier discussion.

## Remaining polish (not done in this pass)

- Reviewer 1's length/verbosity complaint: the introduction and discussion were trimmed somewhat
  in the rewrite, but not aggressively cut -- a further editing pass purely for length is still
  worth doing before submission.
- A handful of reviewer 4's smallest textual nitpicks (a few remaining `\citep`/`\citet` choices,
  "generalization of the t-test" phrasing, "integrating out" terminology) were not individually
  hunted down line-by-line in this pass -- the substantive corrections (terminology, the worked
  example, the figure, the statistics) are done, but a final line-edit against reviewer 4's
  minor-essential-revisions list would still catch a few more.
- Table 1's caption explains the single-hyperparameter-per-dataset design choice (reviewer 3,
  point 12) but the two-sweep model-selection limitation itself is not fixed, only documented.
- `responses.tex` (the point-by-point reviewer response letter) has not been drafted yet -- this
  plan and `findings.md` are the evidence base for it, but the letter itself is a separate,
  not-yet-started deliverable.
