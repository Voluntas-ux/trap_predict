# src

Two implementations, kept side by side so results can be compared directly.

## `trap_controller_inference.py` — the v2 framework

Controller identification as exact Bayesian model comparison, plus prequential prediction. The two questions are answered by one object: the posterior predictive over a library of controller algorithms, updated after every target.

**Model library.** `lam` is contamination strength: 1 is a pure scheme, 0 reduces to the null.

| Family | Mechanism |
|---|---|
| `M0` | independent uniform rows (the null) |
| `A1` | one ordering per station, shared across athletes |
| `A2o` | one cyclic orbit per station, per-athlete offsets free |
| `A4` | within-row constraint: no adjacent repeat (12 of 30 orderings) |
| `A5` | cross-station coupling: an athlete reuses one ordering |
| `A7` | firing-order repeat avoidance (sequential; needs the firing order) |
| `LIB` | scheme library of size K, drawn with replacement across rounds |

All likelihoods are exact. `A1`, `A2o` and `A5` sum over the latent master in closed form; `M0` and `A4` are products over rows; `A7` is a product of sequential conditionals; `LIB` is a product over rounds. Nothing needs approximation: 30 orderings, 6 orbits, 30 rows, 150 targets.

**Prior.** Half the mass on `M0`, the rest split equally across families, uniform over `lam` in {0.25, 0.5, 0.75, 1}. Bayes factors are reported so the prior can be replaced.

**Entry points.**

```python
import trap_controller_inference as C

models, prior, fam = C.build_library()
out = C.run_round(R, models, prior, fam)        # R is a 6x5x5 int array, 0=L 1=S 2=R
post, lbf = C.family_posterior(out["loglik"], prior, fam)

out["hit_mix"]    # per-target hit indicators for the mixture predictor
out["hit_count"]  # the same for the counting baseline
out["ll_mix"]     # per-target log-loss
```

**CLI.**

```bash
python trap_controller_inference.py --demo                 # all three tables
python trap_controller_inference.py --demo --reps 200      # more replications
python trap_controller_inference.py --demo --scen 0,3      # a subset of scenarios
python trap_controller_inference.py --demo --no-lib        # skip the cross-round study
python trap_controller_inference.py --demo --no-single     # only the cross-round study
```

About 0.13 s per round. `--demo` also prints the predecessor battery's rejection rate on the same rounds, which requires `single_round_independence_test.py` to be importable — it is, from this directory.

**Note on λ granularity.** Under weak contamination the family-level Bayes factor is dragged down by the Occam penalty on the high-λ members, so report the per-λ profile alongside the family figure. `--demo` prints both.

**Deliberately omitted.** The predecessor's `A3` column-balancing alternative is a rejection-sampling construction rather than an algorithm a controller would implement. Its physically motivated analogue is `A7`. If an exponential-family tilt is wanted instead, its normaliser is data-independent and estimable once by Monte Carlo.

## `single_round_independence_test.py` — the predecessor battery

Unmodified. Six statistics, each targeted at a different failure mode, combined by calibrated min-p against a Monte Carlo null.

```bash
python single_round_independence_test.py --demo                  # calibration and power
python single_round_independence_test.py --file round.csv         # test one observed round
python single_round_independence_test.py --demo --nnull 20000 --nsim 3000
```

CSV format: one row per athlete-station, columns `athlete,station,v1,v2,v3,v4,v5`, entries `L`, `S`, `R`.

**Two cautions when using it,** both from the report:

1. The p-value floor is `1/(nnull+1)`. When a statistic is pinned there, `argmin(ps)` is not a meaningful attribution — it is decided by the one-sided versus two-sided convention. Read the standardised deviations instead, or use the Bayesian identification.
2. Its `S3` null variance is 625/6 exactly. Do not substitute a chi-square reference; it would understate the spread by eight per cent.
