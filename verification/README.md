# verification

Scripts and logs behind every number in `docs/verification-and-v2.md`. Run `bash run_all.sh` to regenerate the logs; individual scripts can also be run directly.

| Script | Runtime | What it establishes |
|---|---|---|
| `verify_moments.py` | ~30 s | Closed forms for five of six null moments; a 2,000,000-station Monte Carlo of the column statistic; the exact Cov = 1/6 by enumeration; the orbit occupancy behind the A4 attribution split |
| `size_check.py` | ~90 s | Empirical size of the predecessor battery at α = 0.01, 0.05, 0.10 from 20,000 independent test rounds |
| `verify_pooling.py` | ~10 min | The pooling method inferred from the original note, its own size, and its power against A2 and A3 |

## Logs

| Log | Produced by |
|---|---|
| `demo_8000_600.txt` | `single_round_independence_test.py --demo --nnull 8000 --nsim 600` — exact reproduction of the original design note at its own settings |
| `demo_20000_3000_seed7.txt`, `..._part2.txt` | Independent replication at seed 7, 20,000-row null table, 3,000 rounds per cell |
| `size_check.txt` | `size_check.py` |
| `pooling_check.txt` | `verify_pooling.py` (the A3 λ=0.5 row was run separately after an interrupted process; noted inline) |
| `verify_moments.txt` | `verify_moments.py` |
| `proto_single.txt` | `trap_controller_inference.py --demo --reps 80 --no-lib` — identification and prequential prediction |
| `proto_lib.txt` | `trap_controller_inference.py --demo --reps 60 --no-single` — cross-round scheme reuse |
| `lnbf_per_round.txt` | Expected log Bayes factor per round, 80 rounds per truth, seed 11 |

## Reading the logs

Monte Carlo figures will differ in the last digit or two between runs at different seeds; the report quotes the seeds used. Two comparisons are worth making explicitly:

- `demo_8000_600.txt` against the original design note's tables — these should agree **exactly**, and they do. That is what establishes the script as the generator of record.
- `demo_20000_3000_seed7.txt` against the same tables — these agree within Monte Carlo error at a different seed and a larger simulation count, which is what establishes that the original figures were not a lucky draw.
