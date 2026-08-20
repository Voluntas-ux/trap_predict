# Trap controller audit

**Does a clay-target release controller draw each athlete's station sequence independently, and if it does not, what should you predict instead?**

This repository contains a verification of an existing single-round independence test, a replacement framework that answers the prediction question as well as the testing one, and an interactive site that walks through both.

Everything here is a simulation study. No real controller data has been analysed, and no claim is made about any particular range, manufacturer or event.

---

## The problem in one paragraph

Six athletes rotate through five stations, visiting each station five times. At each station an athlete receives two left targets, two right targets and one straight, in an order they do not know. One round is therefore a 6 × 5 × 5 array of direction labels — 150 targets, arranged as 30 rows, each row one of the 30 permutations of {L, L, S, R, R}. Because the allocation per station is fixed, an athlete who counts what they have already had beats chance without any model: **60.67 per cent** over a full round, against 33.3 for a blind guess. Every result in this repository is measured against that baseline. The open question is whether the controller's draws are actually independent, and what predictive edge exists if they are not.

## What is here

| Path | What it is |
|---|---|
| `docs/verification-and-v2.md` | The full report: what was checked, five corrections, the replacement framework, and the limits |
| `docs/source-context.md` | What the prior design note and work plan specify, so the repository stands alone |
| `src/trap_controller_inference.py` | The v2 prototype: exact Bayesian model comparison over a library of controller algorithms, plus prequential prediction |
| `src/single_round_independence_test.py` | The predecessor six-statistic battery, unmodified, kept for comparison and imported by the prototype |
| `verification/` | Scripts and logs that regenerate every number quoted in the report |
| `site/index.html` | A self-contained interactive walkthrough. Open it in a browser; no build step, no server |

## Quick start

```bash
pip install -r requirements.txt

# identification, prediction and cross-round scheme reuse
python src/trap_controller_inference.py --demo

# the predecessor battery: calibration and power
python src/single_round_independence_test.py --demo

# test one observed round
python src/single_round_independence_test.py --file round.csv

# regenerate every verification log
bash verification/run_all.sh
```

To read the site, open `site/index.html` directly, or publish the `site/` directory with GitHub Pages.

## Headline results

**The predecessor test is sound and reproducible.** Every figure in its design note reproduces exactly at its stated settings. Independent replication at a different seed with 25× the simulation count agrees throughout. A 20,000-round calibration check gives empirical size 0.0480 at a nominal 0.05. Five of the six null moments match closed forms to three decimals.

**Five things needed correcting.** The "25 independent columns" argument is false — visit-columns at a station share rows, with covariance exactly 1/6, so the null variance of the column statistic is 625/6 rather than what independence implies. Attribution of the "driving statistic" turns out to be an artefact of the Monte Carlo p-value floor and flips when the null table grows. The pooling table had no script behind it. The summary claim of "power 1.000 against all alternatives" is a claim about five alternatives chosen by the same author who chose the statistics. And the data protocol needs to record firing order.

**The replacement is a posterior, not a p-value.** Seven controller families, exact likelihoods, contamination strength λ as a nuisance parameter, combined into one posterior predictive that is updated after every pull. At full strength it identifies the right family with probability 1.00 and log Bayes factors of 26 to 84. On a clean controller it costs 0.003 nats per target against pure counting — insurance is nearly free. Against a master-plus-offset controller it reaches 91 per cent accuracy by the third block of thirty targets.

**And it still cannot be turned into a precise prediction system.** If the controller is clean, 60.7 per cent is a theorem and there is nothing to find. Weak contamination needs four to nine rounds at the same range to detect and buys one or two accuracy points. Direction class is not a trajectory. Nothing is stationary across sessions, days or ranges. The full list is in the report and in the limitations section of the site.

## Data format

A round is a CSV with one row per athlete-station:

```
athlete,station,v1,v2,v3,v4,v5
1,1,L,R,S,L,R
1,2,R,L,L,R,S
...
```

Entries are `L`, `S` or `R`. Rows that are not permutations of {L, L, S, R, R} fail validation rather than being tested, because a violation means the null is misspecified rather than merely false.

For the Bayesian tool, pass a 6 × 5 × 5 integer array (`0 = L, 1 = S, 2 = R`) to `run_round`, together with the firing order. Under the standard six-athlete rotation, shot `n` is athlete `n mod 6` at station `n mod 5` on visit `n // 30`.

## Scope

This is an offline audit and training-analysis instrument. It measures how much predictive information a release controller leaks, in accuracy points above the counting baseline, computed after the fact from a log. It is not designed for, and should not be used as, an aid during competition.

## Reproducibility

Every table in `docs/verification-and-v2.md` is regenerable by a single command, with seeds recorded in the logs. The site reimplements the generators, the model library and the battery in JavaScript; its live figures agree with the Python reference to the precision shown (log Bayes factors 83.6, 43.4, 26.1, 80.2 in both).

## Licence

No licence has been chosen yet. Add one before publishing if you intend others to reuse the code.
