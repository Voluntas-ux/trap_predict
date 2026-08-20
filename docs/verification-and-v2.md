# Single-Round Independence Diagnostic: Verification Report and a v2 Framework

**Version** 1.1
**Date** 20 August 2026
**Verifies** the single-round independence diagnostic (`src/single_round_independence_test.py`) and its design note, and item W1.4 of the project work plan (summarised in `docs/source-context.md`)
**Delivers** `src/trap_controller_inference.py` (v2 prototype), `verification/` (scripts and run logs that regenerate every number below), and `site/index.html` (an interactive walkthrough of the same material)

---

## 1. Verdict

The implementation is correct, the reported numbers are reproducible to the last digit at the stated settings, the test is correctly calibrated, and five of the six null moments agree with closed forms to three decimals. The note is therefore sound as a piece of engineering. It has one false structural claim (the "25 independent columns" argument, which is repeated verbatim in the handoff), one procedure that does not do what it says (attribution of the driving statistic), one table with no code behind it (pooling), and a framing that is materially too strong in the handoff summary. More importantly, the test is blind by construction to the single most consequential controller behaviour for prediction (scheme reuse across rounds) and weak against firing-order constraints. Section 5 proposes a framework that subsumes the test, removes these gaps, and delivers the prediction the project actually wants.

---

## 2. What was checked and how

**2.1 Exact reproduction.** Running `--demo --nnull 8000 --nsim 600` with the script's default seed reproduces every figure in the note's Sections 3.1 to 3.3 exactly, including the attribution split for A4 (0.46 / 0.47). The note was therefore produced at those settings, and the script is the generator of record for those tables.

**2.2 Independent replication** (seed 7, 20,000-row null table, 3,000 rounds per cell):

| Alternative | lam 0.25 | lam 0.50 | lam 0.75 | lam 1.00 | Note (600 rounds) |
|---|---|---|---|---|---|
| A1 shared sequence | 0.434 | 0.989 | 1.000 | 1.000 | 0.472 / 0.980 / 1.000 / 1.000 |
| A2 master plus offset | 0.131 | 0.605 | 0.992 | 1.000 | 0.102 / 0.605 / 0.993 / 1.000 |
| A3 column balancing | 0.060 | 0.124 | 0.680 | 1.000 | 0.057 / 0.130 / 0.633 / 1.000 |
| A4 no adjacent repeat | 0.175 | 0.700 | 0.994 | 1.000 | 0.148 / 0.685 / 0.990 / 1.000 |
| A5 cross-station coupling | 0.362 | 0.979 | 1.000 | 1.000 | 0.370 / 0.968 / 1.000 / 1.000 |

All within Monte Carlo error of the note. Power at lambda = 1 is 1.000 for the five simulated alternatives, as claimed.

**2.3 Calibration.** The note's size check used 600 rounds, which only bounds the size to roughly 0.036 to 0.072. Rerun with an independent 20,000-row null table and 20,000 test rounds: empirical size 0.0480 (SE 0.0015) at alpha 0.05, 0.0087 at 0.01, 0.0998 at 0.10. The test is calibrated, and slightly conservative at 5 per cent as the discreteness of the statistics predicts. (`verification/size_check.py`)

**2.4 Null moments against closed forms.**

| Statistic | Note (8,000 rounds) | Closed form | Method |
|---|---|---|---|
| S1 | 2.48 (1.54) | 2.5000 (1.5546) | 75 pairs, P = 1/30, pairwise covariances vanish |
| S2 | 19.95 (1.74) | 19.9531 (1.7400) | occupancy of 6 boxes by 6 balls, per station |
| S3 | 50.02 (10.07) | 50 (10.2062) | see 3.1 below; exact Var = 625/6 |
| S4 | 23.95 (4.10) | 24.0000 (4.0988) | exact distribution over the 30 orderings |
| S5 | 2.00 (1.39) | 2.0000 (1.3904) | 60 pairs, P = 1/30 |
| S6 | 10.06 (4.19) | 10 (4.1833) | 5 independent columns, per-column Var = 7/2 |

The per-column Pearson statistic for Multinomial(6, (0.4, 0.2, 0.4)) has exact variance 2(k-1) + (sum 1/p_i - k^2 - 2k + 2)/n = 4 - 1/2 = 3.5, not the asymptotic 4. (`verification/verify_moments.py`)

**2.5 Pooling.** The note's Section 4 table has no implementation. Pooling by summing the six statistics over k rounds and calibrating against a pooled null (sum of k independent null rounds, min-p recombined) reproduces it within Monte Carlo error, so that is presumably what was done:

| Case | k = 1 | k = 2 | k = 3 | k = 5 | Note |
|---|---|---|---|---|---|
| A2 lam 0.25 | 0.135 | 0.158 | 0.225 | 0.328 | 0.116 / 0.176 / 0.200 / 0.352 |
| A2 lam 0.50 | 0.627 | 0.885 | 0.980 | 1.000 | 0.644 / 0.904 / 0.992 / 1.000 |
| A3 lam 0.25 | 0.053 | 0.098 | 0.068 | 0.068 | 0.092 / 0.042 / 0.075 / 0.058 |
| A3 lam 0.50 | 0.120 | 0.207 | 0.313 | 0.500 | 0.092 / 0.225 / 0.325 / 0.508 |

(`verification/verify_pooling.py`; 400 reps for A2 cells, 133 to 150 for A3.)

**2.6 The load-bearing structural assumption.** The work plan flagged the 2 left / 2 right / 1 straight allocation per station as a fact the whole urn model rests on, and as one taken from secondary descriptions rather than a primary source. It is consistent across every published description of the discipline consulted, and the `validate_round` guard in the script fails loudly on any row that violates it, which is the right engineering response: a violation means the null is misspecified rather than merely false. Two further points shape everything below. First, no published description of the discipline says how the per-athlete sequence is generated, whether athletes' sequences are drawn independently, or how a release controller's randomisation is certified. Independence is therefore a modelling assumption with no external support, which is exactly why testing it is the blocking item. Second, the intended use of anything built here is offline: training analysis, range and equipment auditing, and the research pipeline. Nothing in this report is designed for use by an athlete during competition. Section 5.7 builds on this.

---

## 3. Findings that need correction

Ordered by consequence.

**3.1 "25 independent columns" is false.** The note and the handoff both state that one round supplies 25 independent columns of six. The six athletes within a column are independent, which is correct. The five columns at the same station are not: they are built from the same six rows, and each row is a permutation. The columns are positively dependent in the chi-square: Cov(X2_v, X2_v') = 1/6 exactly (enumeration, `verification/verify_moments.py`), so Var(S3) = 5 x (5 x 7/2 + 20 x 1/6) = 625/6 and the null SD is 10.21, not the 9.35 that independence would give. The note's 10.07 came from simulation and is right; the reasoning offered for it is wrong. Nothing in the test breaks, because p-values come from the Monte Carlo null, but (i) the rationale in W1.4 must be reworded to "five independent station blocks, within which athletes are independent but visit-columns are not", and (ii) no analytic shortcut for S3 may be used: a chi-square(50) reference would understate the null spread by 8 per cent and be anticonservative in the upper tail.

**3.2 Attribution is an artefact of the p-value floor.** The note reports that A4 splits between S2 (0.46) and S4 (0.47) and explains it by thinning of the orbit space. The thinning is real: the 12 clean orderings occupy 4 of the 6 orbits with weights 5/12, 5/12, 1/12, 1/12, so E[S2] under A4 is 13.7 against 19.95. But the split is not about relative sensitivity. Under A4 at lambda = 1, S4 is identically 0, a value never seen in 8,000 null rounds (null minimum 10), while S2 is about 13.5, which the same table also never reaches (null minimum 14). Both p-values then sit at their floors, but S4 is declared two-sided and pays a factor of two (2/8001) while S2 is one-sided (1/8001), so `argmin` credits S2 whenever S2 is at the floor, which happens in half the A4 rounds. With a 20,000-row table the same procedure gives S2 0.22 / S4 0.77. The "driving statistic" is therefore a function of table size and sidedness conventions precisely in the regime where rejection is strongest and attribution matters most. Step 3 of the W1.4 procedure ("read the driving statistic, adopt the corresponding structural model") should not be executed on floored p-values. Report standardised deviations from the null mean alongside the p-values, or use the likelihood-based identification in Section 5, which has no floor.

**3.3 Pooling is not regenerable.** W9 requires every table to be regenerable by a single script; the Section 4 pooling table is not. The method inferred in 2.5 should be added to the script as `--pool k`.

**3.4 The handoff's summary over-claims.** "Power is 1.000 from a single round against all five structured alternatives" is true and is also a statement about five alternatives chosen by the author, one of which (A3, best-of-400 rejection towards balanced columns) is a construction rather than anything a controller would plausibly implement. The note's own Section 4 is careful about this; the handoff's W1.4 paragraph is not. Section 4 below quantifies two plausible alternatives against which the battery is weak or blind.

**3.5 Minor.** `_minp_null` is documented as leave-one-out but includes each row in its own reference set; the discrepancy is O(1/n) and the 20,000-round size check shows it is harmless. S6 is a strict subset of S3 (the visit-1 columns), so the two are positively dependent under the null and S6 adds nothing unless opening-slot effects are specific. The note's remark that machine identity would require a 15-symbol alphabet is superfluous: within a station, machine role and direction class are the same thing under the 2/2/1 rule, so the 3-symbol analysis conditional on station already is the machine-level analysis. The data protocol should record firing order (or derive it as shot n -> athlete n mod 6, station n mod 5, visit floor(n/30) for a full squad under the standard rotation), because the alternatives that matter most need it and irregular-target repeats have to be reconciled against it.

---

## 4. Limits the note understates

**4.1 Scheme reuse across rounds.** A controller that holds a library of K fixed full-round schemes and draws one per round is, within any single round, exactly H0-distributed, so no single-round statistic can have power against it. Simulated with K = 20, the battery rejects 5.4 per cent of library rounds, which is its size. This is not an exotic alternative. The vocabulary of the discipline is one of schemes: equipment and discipline descriptions routinely say that targets are selected by a shooting scheme, and at least one description of the five-machine variant, which uses the same selection computer, describes the round sequences as predetermined but unknown to the squad. None of this proves reuse at any particular range; it shows the alternative is the natural one to test first. It is also the alternative with by far the largest predictive payoff, because once a scheme repeats it is identified from its first few targets and the rest of the round is known. The multi-round uniformity test in W1.2 would not catch it either, since a library of randomly generated schemes is uniform over the 30 orderings in the long run.

**4.2 Firing-order constraints.** A controller that draws the next target sequentially and avoids repeating the direction just thrown from the same trap group produces column under-dispersion, which S3 sees only indirectly. The battery rejects 78 per cent of such rounds at full strength and 14 per cent at half strength, against 100 per cent and 73 per cent posterior identification by the framework in Section 5 (Table 5.3). A test built on the firing-order likelihood sees it directly; a battery of marginal statistics does not.

**4.3 Weak contamination.** The note is right that lambda near 0.25 is not detectable in one round and that summing statistics across rounds helps slowly. Section 5.5 replaces the pooling table with an exact calculus: the expected log Bayes factor per round under each alternative, which tells you how many rounds at the same range are needed for decisive evidence without running a new power study per design.

---

## 5. The v2 framework: controller identification and prequential prediction

**5.1 Reframing.** The project's question is not "can independence be rejected" but "what is the probability distribution of the next target, given everything the squad has seen, and does anything beat counting". A significance test answers the first and is silent on the second. The natural object is the posterior predictive over an explicit library of controller algorithms, updated after every pull. It answers both questions with one computation: the posterior over algorithms is the identification result (graded, with Bayes factors instead of a reject/accept at a floor), and the predictive is the prior that W6 needs and the metric that W7 needs. Under the null it collapses to the counting predictor and the 60.7 per cent ceiling; under any structure it does better, and the difference is measured shot by shot on held-out rounds. That prequential score is the only test that matters for a predictor, and it is robust to the library being incomplete in a way a test is not: a model that is wrong but predicts better than counting is still evidence of exploitable structure.

**5.2 Library and likelihoods.** Write r_as in Omega (|Omega| = 30) for the row of athlete a at station s, and M_as for the set of orderings consistent with what has been observed of that row so far (a length-30 mask). Every model below has an exact prefix likelihood and an exact next-target predictive, obtained by updating one mask and renormalising. Contamination lambda mixes the scheme with an independent uniform draw, so lambda = 0 is H0 for every family.

- M0, independent uniform rows: P(prefix) = prod_as |M_as| / 30.
- A1, one ordering per station shared across athletes: per station (1/30) sum_m prod_a [lambda 1(m in M_as) + (1 - lambda)|M_as|/30].
- A2o, one cyclic orbit per station, offsets free: per station (1/6) sum_j prod_a [lambda |M_as ∩ O_j|/5 + (1 - lambda)|M_as|/30], O_j the six orbits. This generalises the note's A2, whose offsets were fixed to a mod 5.
- A4, no adjacent repeat: per row lambda |M_as ∩ C|/12 + (1 - lambda)|M_as|/30, C the 12 clean orderings.
- A5, athlete reuses one ordering across stations: per athlete (1/30) sum_b prod_s [lambda 1(b in M_as) + (1 - lambda)|M_as|/30].
- A7, firing-order repeat avoidance: sequential; the next target for the shooter is drawn from the remaining urn with the direction just thrown from the same group down-weighted by (1 - lambda); likelihood is the product of these conditionals in firing order.
- LIB, scheme library of size K drawn with replacement: across rounds, a round that exactly repeats an earlier one has likelihood 1/K against 30^-30 under M0, so one exact repeat carries ln BF = 30 ln 30 - ln K = 102 - ln K. Within a round, the predictive mixes the known schemes consistent with the prefix against the M0 continuation.

Prior: 1/2 on M0, the remainder split equally across families, uniform over lambda in {0.25, 0.5, 0.75, 1} within a family. Bayes factors are reported so that the prior can be replaced. Pooling across rounds at the same range is automatic and optimal: likelihoods multiply, log Bayes factors add. The note's A3 is omitted deliberately; it is a sampling construction rather than an algorithm, and its physically motivated analogue (a controller balancing machine usage in firing order) is A7. If an exponential-family tilt exp(-beta X2) is wanted, its normaliser is data-independent and estimable once by Monte Carlo; only its shot-by-shot predictive requires summing over completions.

**5.3 Identification from one round** (`trap_controller_inference.py --demo`, 80 rounds per scenario). Posterior mass by family after 150 shots, median ln BF of the true family against M0, and the original battery's rejection rate on the same rounds:

| Truth | M0 | A1 | A2o | A4 | A5 | A7 | ln BF (true) | min-p battery |
|---|---|---|---|---|---|---|---|---|
| H0 independent | 0.84 | 0.03 | 0.04 | 0.03 | 0.03 | 0.04 | 1.1 (for M0) | 0.03 |
| A1 lam 1 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 83.6 | 1.00 |
| A1 lam 0.5 | 0.00 | 0.99 | 0.00 | 0.00 | 0.00 | 0.00 | 16.0 | 1.00 |
| A2 lam 1 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 43.4 | 1.00 |
| A2 lam 0.5 | 0.20 | 0.00 | 0.75 | 0.02 | 0.00 | 0.03 | 4.3 | 0.64 |
| A2 lam 0.25 | 0.72 | 0.02 | 0.15 | 0.03 | 0.02 | 0.05 | -1.1 | 0.12 |
| A4 lam 1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 26.1 | 1.00 |
| A4 lam 0.5 | 0.18 | 0.01 | 0.02 | 0.77 | 0.01 | 0.01 | 3.3 | 0.57 |
| A5 lam 1 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 80.2 | 1.00 |
| A5 lam 0.5 | 0.03 | 0.00 | 0.00 | 0.01 | 0.96 | 0.00 | 14.3 | 0.97 |
| A7 lam 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 46.2 | 0.78 |
| A7 lam 0.5 | 0.23 | 0.00 | 0.01 | 0.01 | 0.01 | 0.73 | 3.5 | 0.14 |

Identification at full strength is exact and diagnostically separating with no floor and no tie-break. At half strength the framework reports graded evidence (ln BF 3 to 16) where the battery's rejection rate is 14 to 100 per cent. At lambda = 0.25 the evidence is weak and says so; the family marginal is dragged below 1 by the Occam penalty on the high-lambda members, which is the honest answer and is why the per-lambda profile should be reported alongside the family figure. Under H0 the posterior on M0 rises from its prior of 0.5 to 0.84, and the false rejection rate of the battery is 3 per cent.

**5.4 Prediction** (same run). Accuracy by 30-shot block and mean log-loss per shot, for the counting predictor (M0), the posterior mixture, and the oracle that knows the true model and lambda:

| Truth | Predictor | b1 | b2 | b3 | b4 | b5 | All | Log-loss |
|---|---|---|---|---|---|---|---|---|
| H0 | counting | 0.407 | 0.500 | 0.533 | 0.584 | 1.000 | 0.605 | 0.680 |
| H0 | mixture | 0.390 | 0.500 | 0.532 | 0.599 | 1.000 | 0.604 | 0.684 |
| A1 lam 1 | counting | 0.405 | 0.528 | 0.523 | 0.637 | 1.000 | 0.618 | 0.680 |
| A1 lam 1 | mixture | 0.886 | 0.921 | 0.922 | 0.940 | 1.000 | 0.934 | 0.138 |
| A2 lam 1 | counting | 0.403 | 0.498 | 0.537 | 0.599 | 1.000 | 0.607 | 0.680 |
| A2 lam 1 | mixture | 0.396 | 0.535 | 0.911 | 1.000 | 1.000 | 0.768 | 0.406 |
| A2 lam 0.5 | mixture | 0.394 | 0.504 | 0.605 | 0.768 | 1.000 | 0.654 | 0.655 |
| A4 lam 1 | counting | 0.407 | 0.740 | 0.511 | 0.498 | 1.000 | 0.631 | 0.680 |
| A4 lam 1 | mixture | 0.410 | 0.751 | 0.675 | 0.663 | 1.000 | 0.700 | 0.522 |
| A5 lam 1 | mixture | 0.853 | 0.905 | 0.905 | 0.916 | 1.000 | 0.916 | 0.161 |
| A7 lam 1 | counting | 0.375 | 0.418 | 0.485 | 0.649 | 1.000 | 0.585 | 0.680 |
| A7 lam 1 | mixture | 0.557 | 0.652 | 0.760 | 0.872 | 1.000 | 0.769 | 0.389 |
| A7 lam 0.5 | mixture | 0.450 | 0.511 | 0.560 | 0.693 | 1.000 | 0.643 | 0.668 |

Three things to read off. The cost of insurance under H0 is 0.003 nats per shot and 0.001 in accuracy, so carrying the library when the controller is clean is free. Under A2 the mixture is at 91 per cent in block 3 and 100 per cent from block 4, which is what "squadmate observation yields nothing" in W8 turns into once the structure is learnable. The mixture is within 0.02 of the oracle in accuracy throughout, so the loss from not knowing which algorithm is in play is small compared with the gain from knowing that one is. The full table with oracle rows is in `verification/logs/proto_single.txt`.

**5.5 How many rounds are needed.** Expected ln BF per round of the best-matching member against M0, at the same range, with the rounds needed to reach ln BF = 5 (about 150 to 1):

| Truth | Best member | Mean ln BF per round (sd) | Rounds to ln BF 5 |
|---|---|---|---|
| A2 lam 0.25 | A2o lam 0.25 | 0.57 (1.13) | 8.7 |
| A2 lam 0.50 | A2o lam 0.50 | 6.97 (4.36) | 0.7 |
| A4 lam 0.25 | A4 lam 0.25 | 1.38 (1.79) | 3.6 |
| A7 lam 0.25 | A7 lam 0.25 | 0.80 (1.18) | 6.3 |
| A7 lam 0.50 | A7 lam 0.50 | 4.84 (2.77) | 1.0 |
| H0 | best alternative | -0.50 (0.86) | evidence accrues for M0 |

This replaces the pooling power table: no new simulation per design, a direct reading of the information rate, and a stopping rule (stop when the cumulative ln BF for or against M0 crosses a threshold) instead of a fixed k.

**5.6 Scheme reuse across rounds** (K = 20, 12 rounds, 20 replications). Mean accuracy per round of the counting predictor and of the mixture including LIB, and the cumulative ln BF for LIB against M0:

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| counting | 0.60 | 0.59 | 0.60 | 0.61 | 0.61 | 0.60 | 0.59 | 0.60 | 0.59 | 0.63 | 0.61 | 0.60 |
| mixture | 0.61 | 0.60 | 0.67 | 0.68 | 0.65 | 0.68 | 0.76 | 0.79 | 0.70 | 0.73 | 0.77 | 0.79 |
| cum ln BF | 0 | 0 | 15 | 34 | 44 | 64 | 103 | 148 | 172 | 201 | 241 | 290 |

Each repeat adds about 99 to the log evidence and is identified within the first handful of targets of the repeated round, after which that round is predicted perfectly; the per-round average is diluted by the rounds that are new. The battery's rejection rate on these rounds is 0.054. Matching should also be run up to athlete-slot relabelling and cyclic rotation of slots, since a range may apply the same scheme to whichever squad is present; that is a six-fold or thirty-fold enlargement of the match test, not a change of method. Once a library is detected, a scheme registry keyed by the first few targets becomes the prediction engine for that range.

**5.7 What changes in the handoff.**

- W1.2 stays, but can be cast in the same language: a Bayes factor of Dirichlet-multinomial against uniform over the 30 orderings, accumulated across station-sequences, which removes the fixed 1,100-sequence target and gives a stopping rule.
- W1.4 becomes controller identification: run `trap_controller_inference.py` on every collected round, report the posterior over families and the per-lambda profile, accumulate across rounds at the same range, and run the cross-round match test on every new round. The original battery is kept as a model-free screen, with the corrections in Section 3.
- W1.5's "replacement model is a joint chain on the master permutation plus per-athlete offset" presumes the answer. The replacement model is the posterior mixture, whatever it turns out to be.
- W6's prior is the posterior predictive, which is already the prior-free and prior-informed variants the handoff asks for (M0 alone against the mixture).
- W7's counting baseline is M0's predictive; the headline number for any range becomes the mixture's accuracy over counting on held-out rounds, split by range and session as W7.2 requires.
- W8's squadmate observation is no longer a low-priority extension; it is the inferential channel.
- W1.1 adds firing order and an explicit reconciliation rule for irregular-target repeats.

**5.8 The larger framing.** What this cannot be is a device an athlete uses on the line; the whole design here is offline. What it can be is larger than a prediction tool. The same posterior predictive, scored against counting, is a measure of how much a range's release controller leaks, in accuracy points over 60.7 per cent. Published specifications of the discipline constrain the marginal allocation of targets and are silent on the generation mechanism; none of them describes an audit of a control unit's randomness. A controller that uses a scheme library or a sequential heuristic gives an edge to athletes who notice, which is a fairness problem before it is a prediction opportunity. So the deliverable that scales is an open audit: a fixed data schema (round, squad, athlete, station, visit, shot index, direction, range, date, controller make), the identification engine, the predictability score, and a registry of scheme fingerprints per range. Coaches use it to choose where to train and how to count; ranges use it to certify their controllers; federations have something to point to when a result is questioned; and the vision work in W3 to W6 inherits a prior that has been measured rather than assumed. The open question 1 in the handoff (what algorithm does the controller implement) then has an empirical answer path that does not depend on a manufacturer's cooperation, while a controller log remains the ground truth that makes the test confirmatory rather than exploratory.

---

## 6. Files

- `src/trap_controller_inference.py`: the v2 prototype. `--demo` regenerates Tables 5.3, 5.4 and 5.6 (`--reps`, `--seed`, `--scen`, `--no-single`, `--no-lib`). About 0.13 s per round. It imports `single_round_independence_test.py` (an unchanged copy is included alongside) for the battery comparison column; `run_round` and `family_posterior` are the entry points for real data, which should be supplied as a 6 x 5 x 5 integer array (0 = L, 1 = S, 2 = R) plus the firing order.
- `verification/verify_moments.py`: closed forms, the 2,000,000-station Monte Carlo, the exact 1/6 covariance, and the A4 orbit occupancy.
- `verification/size_check.py`: the 20,000-round calibration check.
- `verification/verify_pooling.py`: the inferred pooling method and its power table.
- `verification/logs/*.txt`: run logs for every number quoted above, including the exact reproduction of the note at its settings and the seed-7 replication.
- `site/index.html`: a self-contained interactive walkthrough. It reimplements the round generators, the seven-family model library and the six-statistic battery in JavaScript, so every claim below can be exercised live rather than read off a table.
- `docs/source-context.md`: what the prior note and work plan specify, for readers who do not have them to hand.
