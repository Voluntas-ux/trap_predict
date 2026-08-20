"""Pooling check. The note's Section 4 pooling table has no implementation in
single_round_independence_test.py. This reproduces the natural version:
statistics are summed over k rounds and calibrated against a pooled null
(sum of k independent null rounds), min-p combined exactly as in the original.
"""
import os, sys
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")]
import sys
import numpy as np
import single_round_independence_test as T

N_NULL = 20000
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
base = T.build_null_table(N_NULL, seed=11)
rng = np.random.default_rng(2024)


def pooled_null(k):
    if k == 1:
        return base
    idx = rng.integers(0, N_NULL, size=(N_NULL, k))
    return base[idx].sum(1)


def pooled_stats(gen, lam, k):
    tot = np.zeros(len(T.STAT_NAMES))
    for _ in range(k):
        R = gen(rng, lam=lam)
        tot += np.array([T.STATS[n][0](R) for n in T.STAT_NAMES])
    return tot


def power(gen, lam, k, table, minp_tab, reps, alpha=0.05):
    hits = 0
    for _ in range(reps):
        obs = pooled_stats(gen, lam, k)
        ps = T._pvals(obs, table)
        gp = (np.sum(minp_tab <= ps.min()) + 1) / (len(minp_tab) + 1)
        hits += gp <= alpha
    return hits / reps


ks = [1, 2, 3, 5]
tables = {k: pooled_null(k) for k in ks}
minps = {k: T._minp_null(tables[k]) for k in ks}

# size check of the pooled machinery
print("size of pooled test under H0 (600 reps each):")
for k in ks:
    print(f"  k={k}: {power(T.gen_null if False else (lambda r, lam=0: T.gen_null(r)), 0, k, tables[k], minps[k], 600):.3f}")

print(f"\npower, {REPS} reps per cell, alternatives A2 and A3:")
print(f"{'case':<26}" + "".join(f"{'k='+str(k):>8}" for k in ks))
for name, gen in [("A2 master+offset", T.gen_master_offset), ("A3 column balancing", T.gen_column_balanced)]:
    for lam in (0.25, 0.5):
        reps = REPS if name.startswith("A2") else max(100, REPS // 3)
        row = [power(gen, lam, k, tables[k], minps[k], reps) for k in ks]
        print(f"{name+' lam='+str(lam):<26}" + "".join(f"{v:>8.3f}" for v in row))
