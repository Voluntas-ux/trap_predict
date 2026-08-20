"""
Single-round independence diagnostic for Olympic Trap target allocation.

Question
--------
Does the target selection controller draw each athlete's station sequence
independently, or is there a shared latent structure (a master permutation with
per-athlete offsets, per-slot balancing, or within-row constraints)?

Why one round is enough
-----------------------
The naive framing counts 25 scalar observations per athlete and concludes that a
single round is hopeless. That framing is wrong for two reasons.

1.  The whole squad is observable. A round of six athletes over five stations
    produces 150 labelled targets arranged as a complete 6 x 5 x 5 array, i.e.
    30 fully observed permutation-rows, not 25 scalars.

2.  The null is FULLY SPECIFIED. Under independence each row is a uniform draw
    from the 30 orderings of {L,L,S,R,R}. There are no nuisance parameters, so
    exact Monte Carlo p-values are available for any statistic. Power, not
    calibration, is the only question.

The single most useful consequence of (2): under the null, the six athlete
outcomes at a fixed (station, visit) cell are exactly six i.i.d. draws from
Multinomial(1, (0.4, 0.2, 0.4)). That gives 25 independent columns of six from a
single round, and any cross-athlete coupling shows up as over- or under-
dispersion in those columns.

Input format
------------
round_array : ndarray of shape (n_athletes=6, n_stations=5, n_visits=5), dtype
object or int, entries in {'L','S','R'}. Element [a, s, v] is the direction
class of the target athlete a received on their v-th visit to station s.

Ground truth should come from the controller log where available. Video coding
of direction class is acceptable; video coding of specific machine ID is not.

Usage
-----
    python single_round_independence_test.py --demo          # calibration + power
    python single_round_independence_test.py --file r.csv    # test a real round

CSV format: one row per (athlete, station), columns
athlete,station,v1,v2,v3,v4,v5
"""

import argparse
import csv
import sys
from itertools import permutations

import numpy as np

LABELS = ("L", "S", "R")
MULTISET = ("L", "L", "S", "R", "R")
ORDERINGS = sorted(set(permutations(MULTISET)))          # 30
N_ORD = len(ORDERINGS)
ORD_INDEX = {o: i for i, o in enumerate(ORDERINGS)}
MARGINAL = np.array([0.4, 0.2, 0.4])                     # P(L), P(S), P(R)

# ---------------------------------------------------------------- structure

def _cyclic_shift(o, k):
    return tuple(o[(i + k) % 5] for i in range(5))


def _build_orbits():
    """Cyclic-shift orbits of the 30 orderings. 5 is prime and no ordering is
    constant, so every orbit has size exactly 5, giving 6 orbits."""
    seen, orbits = set(), []
    for o in ORDERINGS:
        if o in seen:
            continue
        orb = {_cyclic_shift(o, k) for k in range(5)}
        orbits.append(orb)
        seen |= orb
    return orbits


ORBITS = _build_orbits()
ORBIT_OF = {o: i for i, orb in enumerate(ORBITS) for o in orb}
N_ORBIT = len(ORBITS)                                    # 6


# ---------------------------------------------------------------- statistics
#
# Each statistic returns a scalar. Direction of the alternative is recorded in
# STAT_SIDE: 'low' means dependence pushes the statistic DOWN, 'high' means up,
# 'two' means either tail is evidence.

def s1_identical_pairs_within_station(R):
    """Count of athlete pairs sharing an identical ordering at the same station.
    Null expectation = 5 stations * C(6,2) * (1/30) = 2.5.
    Detects a controller that reuses one sequence across athletes."""
    n_a, n_s, _ = R.shape
    c = 0
    for s in range(n_s):
        rows = [tuple(R[a, s]) for a in range(n_a)]
        for i in range(n_a):
            for j in range(i + 1, n_a):
                c += rows[i] == rows[j]
    return float(c)


def s2_distinct_orbits_per_station(R):
    """Number of distinct cyclic orbits occupied by the six athlete rows,
    summed over stations. Under the null this is six balls into six boxes,
    expectation 6*(1-(5/6)^6) = 3.99 per station. A master-plus-offset scheme
    forces every athlete into ONE orbit, driving this to 1 per station."""
    n_a, n_s, _ = R.shape
    tot = 0
    for s in range(n_s):
        tot += len({ORBIT_OF[tuple(R[a, s])] for a in range(n_a)})
    return float(tot)


def s3_column_dispersion(R):
    """Summed Pearson chi-square over the 25 (station, visit) columns, each a
    count vector of six athlete outcomes tested against Multinomial(6, MARGINAL).

    This is the workhorse. Under the null the six entries of a column are exactly
    i.i.d., so the null distribution is known without approximation. Balancing
    schemes drive it DOWN (under-dispersion); clustering drives it UP."""
    n_a, n_s, n_v = R.shape
    exp = n_a * MARGINAL
    tot = 0.0
    for s in range(n_s):
        for v in range(n_v):
            col = R[:, s, v]
            obs = np.array([np.sum(col == lab) for lab in LABELS], dtype=float)
            tot += float(np.sum((obs - exp) ** 2 / exp))
    return tot


def s4_adjacent_repeats(R):
    """Total count of adjacent identical entries across all 30 rows.
    Null expectation = 30 * 0.8 = 24. Detects a within-row constraint such as
    'never present the same direction twice in a row'."""
    n_a, n_s, _ = R.shape
    c = 0
    for a in range(n_a):
        for s in range(n_s):
            row = R[a, s]
            c += int(np.sum(row[:-1] == row[1:]))
    return float(c)


def s5_identical_pairs_within_athlete(R):
    """Count of station pairs sharing an identical ordering for the same athlete.
    Null expectation = 6 athletes * C(5,2) * (1/30) = 2.0.
    Detects cross-station coupling within an athlete."""
    n_a, n_s, _ = R.shape
    c = 0
    for a in range(n_a):
        rows = [tuple(R[a, s]) for s in range(n_s)]
        for i in range(n_s):
            for j in range(i + 1, n_s):
                c += rows[i] == rows[j]
    return float(c)


def s6_first_target_concentration(R):
    """Chi-square of the first-target label counts at each station across the six
    athletes, summed. Isolates the opening slot, where a controller that seeds
    all athletes from one draw is most exposed."""
    n_a, n_s, _ = R.shape
    exp = n_a * MARGINAL
    tot = 0.0
    for s in range(n_s):
        col = R[:, s, 0]
        obs = np.array([np.sum(col == lab) for lab in LABELS], dtype=float)
        tot += float(np.sum((obs - exp) ** 2 / exp))
    return tot


STATS = {
    "S1 identical pairs, within station":      (s1_identical_pairs_within_station, "high"),
    "S2 distinct cyclic orbits per station":   (s2_distinct_orbits_per_station,    "low"),
    "S3 column dispersion (station x visit)":  (s3_column_dispersion,              "two"),
    "S4 adjacent repeats within row":          (s4_adjacent_repeats,               "two"),
    "S5 identical pairs, within athlete":      (s5_identical_pairs_within_athlete, "high"),
    "S6 first-target concentration":           (s6_first_target_concentration,     "two"),
}
STAT_NAMES = list(STATS.keys())


# ---------------------------------------------------------------- generators

def gen_null(rng, n_a=6, n_s=5, n_v=5):
    """H0: every athlete-station row an independent uniform draw over the 30."""
    idx = rng.integers(0, N_ORD, size=(n_a, n_s))
    R = np.empty((n_a, n_s, n_v), dtype=object)
    for a in range(n_a):
        for s in range(n_s):
            R[a, s] = np.array(ORDERINGS[idx[a, s]], dtype=object)
    return R


def gen_master_offset(rng, lam=1.0, n_a=6, n_s=5, n_v=5):
    """A2: one master ordering per station; athlete a receives cyclic shift
    a mod 5 of it with probability lam, else an independent uniform draw.
    lam = 1 is a pure offset scheme, lam = 0 reduces to the null."""
    R = np.empty((n_a, n_s, n_v), dtype=object)
    for s in range(n_s):
        master = ORDERINGS[rng.integers(0, N_ORD)]
        for a in range(n_a):
            if rng.random() < lam:
                row = _cyclic_shift(master, a % 5)
            else:
                row = ORDERINGS[rng.integers(0, N_ORD)]
            R[a, s] = np.array(row, dtype=object)
    return R


def gen_shared_sequence(rng, lam=1.0, n_a=6, n_s=5, n_v=5):
    """A1: the controller reuses ONE ordering per station for every athlete,
    with contamination lam as above. The crudest possible dependence."""
    R = np.empty((n_a, n_s, n_v), dtype=object)
    for s in range(n_s):
        master = ORDERINGS[rng.integers(0, N_ORD)]
        for a in range(n_a):
            row = master if rng.random() < lam else ORDERINGS[rng.integers(0, N_ORD)]
            R[a, s] = np.array(row, dtype=object)
    return R


def gen_column_balanced(rng, lam=1.0, n_a=6, n_s=5, n_v=5, tries=400):
    """A3: the controller balances each firing slot, i.e. it prefers assignments
    where the six athletes at a given (station, visit) receive a spread of
    directions close to the 2.4 / 1.2 / 2.4 marginal. Implemented as rejection
    towards minimum column chi-square with strength lam."""
    R = np.empty((n_a, n_s, n_v), dtype=object)
    exp = n_a * MARGINAL
    for s in range(n_s):
        best, best_x = None, None
        for _ in range(tries):
            cand = [ORDERINGS[i] for i in rng.integers(0, N_ORD, size=n_a)]
            x = 0.0
            for v in range(n_v):
                obs = np.array([sum(1 for r in cand if r[v] == lab) for lab in LABELS], float)
                x += float(np.sum((obs - exp) ** 2 / exp))
            if best_x is None or x < best_x:
                best, best_x = cand, x
        plain = [ORDERINGS[i] for i in rng.integers(0, N_ORD, size=n_a)]
        for a in range(n_a):
            row = best[a] if rng.random() < lam else plain[a]
            R[a, s] = np.array(row, dtype=object)
    return R


def gen_no_adjacent_repeat(rng, lam=1.0, n_a=6, n_s=5, n_v=5):
    """A4: within-row constraint. The controller avoids presenting the same
    direction on consecutive visits to a station. 12 of the 30 orderings qualify."""
    clean = [o for o in ORDERINGS if all(o[i] != o[i + 1] for i in range(4))]
    R = np.empty((n_a, n_s, n_v), dtype=object)
    for a in range(n_a):
        for s in range(n_s):
            pool = clean if rng.random() < lam else ORDERINGS
            R[a, s] = np.array(pool[rng.integers(0, len(pool))], dtype=object)
    return R


def gen_cross_station(rng, lam=1.0, n_a=6, n_s=5, n_v=5):
    """A5: coupling across stations within an athlete, i.e. the athlete's five
    station rows are copies of one draw. Invisible to S1/S2/S3, caught by S5."""
    R = np.empty((n_a, n_s, n_v), dtype=object)
    for a in range(n_a):
        base = ORDERINGS[rng.integers(0, N_ORD)]
        for s in range(n_s):
            row = base if rng.random() < lam else ORDERINGS[rng.integers(0, N_ORD)]
            R[a, s] = np.array(row, dtype=object)
    return R


ALTERNATIVES = {
    "A1 shared sequence":        gen_shared_sequence,
    "A2 master + cyclic offset": gen_master_offset,
    "A3 column balancing":       gen_column_balanced,
    "A4 no adjacent repeat":     gen_no_adjacent_repeat,
    "A5 cross-station coupling": gen_cross_station,
}


# ---------------------------------------------------------------- inference

def build_null_table(n_sim=20000, seed=0):
    """Monte Carlo the joint null distribution of all six statistics.
    Returns an (n_sim, 6) array. Reuse across tests; it does not depend on data."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_sim, len(STAT_NAMES)))
    for i in range(n_sim):
        R = gen_null(rng)
        for j, name in enumerate(STAT_NAMES):
            out[i, j] = STATS[name][0](R)
    return out


def _pvals(obs_vec, null_table):
    """Exact Monte Carlo p-values with the +1 correction, respecting each
    statistic's alternative direction."""
    n = null_table.shape[0]
    ps = np.empty(len(STAT_NAMES))
    for j, name in enumerate(STAT_NAMES):
        side = STATS[name][1]
        col = null_table[:, j]
        if side == "high":
            p = (np.sum(col >= obs_vec[j]) + 1) / (n + 1)
        elif side == "low":
            p = (np.sum(col <= obs_vec[j]) + 1) / (n + 1)
        else:
            hi = (np.sum(col >= obs_vec[j]) + 1) / (n + 1)
            lo = (np.sum(col <= obs_vec[j]) + 1) / (n + 1)
            p = min(1.0, 2 * min(hi, lo))
        ps[j] = p
    return ps


def _minp_null(null_table):
    """Calibrate the min-p combination by leave-one-out against the null table.
    Returns the null distribution of min(p) so the family-wise error rate is
    controlled exactly rather than by a conservative Bonferroni bound."""
    n = null_table.shape[0]
    mins = np.empty(n)
    for j, name in enumerate(STAT_NAMES):
        col = null_table[:, j]
        order = np.argsort(col)
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        ties_le = np.searchsorted(np.sort(col), col, side="right")
        ties_ge = n - np.searchsorted(np.sort(col), col, side="left")
        side = STATS[name][1]
        if side == "high":
            p = (ties_ge + 1) / (n + 1)
        elif side == "low":
            p = (ties_le + 1) / (n + 1)
        else:
            p = np.minimum(1.0, 2 * np.minimum((ties_ge + 1) / (n + 1),
                                               (ties_le + 1) / (n + 1)))
        mins = p if j == 0 else np.minimum(mins, p)
    return mins


def test_round(R, null_table, minp_null):
    """Run the full battery on one observed round."""
    obs = np.array([STATS[name][0](R) for name in STAT_NAMES])
    ps = _pvals(obs, null_table)
    minp = ps.min()
    global_p = (np.sum(minp_null <= minp) + 1) / (len(minp_null) + 1)
    return obs, ps, minp, global_p


def validate_round(R):
    """Every athlete-station row must be a permutation of {L,L,S,R,R}.
    A violation means either a coding error or a rule violation, and either way
    the null is not the right null. Fail loudly."""
    n_a, n_s, _ = R.shape
    bad = []
    for a in range(n_a):
        for s in range(n_s):
            if tuple(sorted(R[a, s])) != tuple(sorted(MULTISET)):
                bad.append((a, s, tuple(R[a, s])))
    return bad


# ---------------------------------------------------------------- reporting

def run_power_study(n_null=20000, n_sim=2000, seed=1, alpha=0.05):
    print("Building null table ({} draws) ...".format(n_null))
    null_table = build_null_table(n_null, seed=seed)
    minp_null = _minp_null(null_table)

    print("\n=== NULL CALIBRATION (statistic means under H0) ===")
    for j, name in enumerate(STAT_NAMES):
        col = null_table[:, j]
        print(f"  {name:<42} mean {col.mean():8.3f}   sd {col.std():7.3f}")

    rng = np.random.default_rng(seed + 99)
    print(f"\n=== SIZE CHECK: rejection rate under H0 at alpha={alpha} "
          f"({n_sim} simulated rounds) ===")
    rej = 0
    for _ in range(n_sim):
        R = gen_null(rng)
        _, _, _, gp = test_round(R, null_table, minp_null)
        rej += gp <= alpha
    print(f"  empirical size = {rej / n_sim:.4f}   (target {alpha})")

    print(f"\n=== POWER FROM A SINGLE ROUND, alpha={alpha}, {n_sim} rounds per cell ===")
    lams = [0.25, 0.50, 0.75, 1.00]
    header = f"{'alternative':<28}" + "".join(f"{f'lam={l:.2f}':>10}" for l in lams)
    print(header)
    results = {}
    for aname, gen in ALTERNATIVES.items():
        row = []
        for lam in lams:
            hits = 0
            n_here = n_sim if aname != "A3 column balancing" else max(300, n_sim // 6)
            for _ in range(n_here):
                R = gen(rng, lam=lam)
                _, _, _, gp = test_round(R, null_table, minp_null)
                hits += gp <= alpha
            row.append(hits / n_here)
        results[aname] = row
        print(f"{aname:<28}" + "".join(f"{v:>10.3f}" for v in row))

    print("\n=== WHICH STATISTIC CARRIES EACH ALTERNATIVE (lam=1.0, 400 rounds) ===")
    print(f"{'alternative':<28}" + "".join(f"{n.split()[0]:>7}" for n in STAT_NAMES))
    for aname, gen in ALTERNATIVES.items():
        counts = np.zeros(len(STAT_NAMES))
        n_here = 400 if aname != "A3 column balancing" else 150
        for _ in range(n_here):
            R = gen(rng, lam=1.0)
            _, ps, _, _ = test_round(R, null_table, minp_null)
            counts[np.argmin(ps)] += 1
        print(f"{aname:<28}" + "".join(f"{v/n_here:>7.2f}" for v in counts))

    return null_table, minp_null, results


def load_csv(path):
    rows = {}
    with open(path) as f:
        for rec in csv.DictReader(f):
            a = int(rec["athlete"]) - 1
            s = int(rec["station"]) - 1
            rows[(a, s)] = [rec[f"v{i}"].strip().upper() for i in range(1, 6)]
    n_a = max(k[0] for k in rows) + 1
    n_s = max(k[1] for k in rows) + 1
    R = np.empty((n_a, n_s, 5), dtype=object)
    for (a, s), v in rows.items():
        R[a, s] = np.array(v, dtype=object)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="calibration and power study")
    ap.add_argument("--file", help="CSV of one observed round")
    ap.add_argument("--nnull", type=int, default=20000)
    ap.add_argument("--nsim", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    if args.demo:
        run_power_study(n_null=args.nnull, n_sim=args.nsim, alpha=args.alpha)
        return

    if not args.file:
        ap.error("pass --demo or --file")

    R = load_csv(args.file)
    bad = validate_round(R)
    if bad:
        print("ROW VALIDATION FAILED. These rows are not permutations of LLSRR:")
        for b in bad:
            print("   athlete {} station {}: {}".format(b[0] + 1, b[1] + 1, b[2]))
        print("Resolve before testing. The null assumes the 2/1/2 allocation holds.")
        sys.exit(1)

    null_table = build_null_table(args.nnull)
    minp_null = _minp_null(null_table)
    obs, ps, minp, gp = test_round(R, null_table, minp_null)

    print("=== SINGLE-ROUND INDEPENDENCE DIAGNOSTIC ===\n")
    print(f"{'statistic':<42}{'observed':>10}{'null mean':>11}{'p':>9}")
    for j, name in enumerate(STAT_NAMES):
        print(f"{name:<42}{obs[j]:>10.3f}{null_table[:, j].mean():>11.3f}{ps[j]:>9.4f}")
    print(f"\nmin p = {minp:.4f}")
    print(f"family-wise p (min-p, calibrated) = {gp:.4f}")
    print("\nInterpretation:")
    if gp <= 0.05:
        print("  Independence is rejected at the 5 per cent level from this single round.")
        print("  The driving statistic is:", STAT_NAMES[int(np.argmin(ps))])
        print("  The Section 7 factorisation in the report must be rebuilt.")
    else:
        print("  No evidence against independence from this round. Note that a single")
        print("  round has low power against weak contamination; see the power table")
        print("  in the demo output before treating this as confirmation.")


if __name__ == "__main__":
    main()
