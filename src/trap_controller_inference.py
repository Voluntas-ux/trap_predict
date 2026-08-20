"""
Controller identification for Olympic Trap as exact Bayesian model
comparison plus prequential prediction.

Replaces "is independence rejected?" with two questions that are closer to the
project's purpose:

  1. Which controller algorithm, from an explicit library, generated this data,
     and with what posterior probability?  (exact likelihoods, no Monte Carlo
     null, no p-value floor, graded evidence via Bayes factors)
  2. How well can the next target be predicted, shot by shot, using everything
     the squad has seen so far, and does anything beat plain counting?
     (prequential log-loss and accuracy; the only test that matters for a
     predictor)

Both questions are answered by the same object: the posterior predictive
mixture over the model library, updated after every pull.

Model library (lam = contamination strength, 1 = pure scheme, 0 = H0)
  M0   independent uniform rows                                  (the null)
  A1   one ordering per station shared across athletes
  A2o  one cyclic orbit per station, per-athlete offset free     (generalises A2)
  A4   within-row constraint: no adjacent repeat (12/30 orderings)
  A5   cross-station coupling: athlete reuses one ordering
  A7   firing-order constraint: controller avoids repeating the direction
       just thrown from the same trap group (sequential, needs firing order)
  LIB  scheme reuse across rounds: a fixed library of K full-round schemes
       drawn with replacement (invisible within a round, decisive across rounds)

All likelihoods are exact; A1/A2o/A5 sum over the latent master in closed form,
M0/A4 are products over rows, A7 is a product of sequential conditionals, LIB
is a product over rounds.  Everything is small enough that no approximation
is needed: 30 orderings, 6 orbits, 30 rows, 150 shots.

Usage
  python trap_controller_inference.py --demo [--reps 60]
"""
import argparse
import sys
from itertools import permutations

import numpy as np

# ---------------------------------------------------------------- structure
LABELS = ("L", "S", "R")
LAB = {c: i for i, c in enumerate(LABELS)}
ORD_T = sorted(set(permutations("LLSRR")))                  # 30 tuples
ORD = np.array([[LAB[c] for c in o] for o in ORD_T])         # 30 x 5 ints
N_ORD = 30
ORD_INDEX = {o: i for i, o in enumerate(ORD_T)}


def _shift(o, k):
    return tuple(o[(i + k) % 5] for i in range(5))


_orbits, _seen = [], set()
for _o in ORD_T:
    if _o in _seen:
        continue
    _orb = {_shift(_o, k) for k in range(5)}
    _orbits.append(_orb)
    _seen |= _orb
ORBIT = np.zeros((6, N_ORD), dtype=bool)                     # orbit j x ordering
for j, orb in enumerate(_orbits):
    for o in orb:
        ORBIT[j, ORD_INDEX[o]] = True
CLEAN = np.array([all(o[i] != o[i + 1] for i in range(4)) for o in ORD_T])   # 12 True
POS_LAB = np.zeros((5, 3, N_ORD), dtype=bool)                # visit v, label x -> orderings with o[v]==x
for v in range(5):
    for x in range(3):
        POS_LAB[v, x] = ORD[:, v] == x

# firing order: shot n is athlete n mod 6 at station n mod 5, visit n // 30
FIRING = [(n % 6, n % 5, n // 30) for n in range(150)]

# ---------------------------------------------------------------- generators
def gen_null(rng):
    idx = rng.integers(0, N_ORD, size=(6, 5))
    return ORD[idx].copy()                                   # 6 x 5 x 5 ints


def gen_shared(rng, lam):
    R = np.empty((6, 5, 5), dtype=int)
    for s in range(5):
        m = rng.integers(0, N_ORD)
        for a in range(6):
            R[a, s] = ORD[m] if rng.random() < lam else ORD[rng.integers(0, N_ORD)]
    return R


def gen_master_offset(rng, lam, free_offsets=False):
    R = np.empty((6, 5, 5), dtype=int)
    for s in range(5):
        m = ORD_T[rng.integers(0, N_ORD)]
        for a in range(6):
            if rng.random() < lam:
                k = rng.integers(0, 5) if free_offsets else a % 5
                R[a, s] = [LAB[c] for c in _shift(m, k)]
            else:
                R[a, s] = ORD[rng.integers(0, N_ORD)]
    return R


def gen_no_adjacent(rng, lam):
    clean_idx = np.flatnonzero(CLEAN)
    R = np.empty((6, 5, 5), dtype=int)
    for a in range(6):
        for s in range(5):
            pool = clean_idx if rng.random() < lam else np.arange(N_ORD)
            R[a, s] = ORD[pool[rng.integers(0, len(pool))]]
    return R


def gen_cross_station(rng, lam):
    R = np.empty((6, 5, 5), dtype=int)
    for a in range(6):
        b = rng.integers(0, N_ORD)
        for s in range(5):
            R[a, s] = ORD[b] if rng.random() < lam else ORD[rng.integers(0, N_ORD)]
    return R


def gen_firing_order_avoid(rng, lam):
    """A7: in firing order, the controller draws the next target for the shooter
    from the shooter's remaining urn at that station, down-weighting the
    direction that was just thrown from the same trap group.  lam = 1 is strict
    avoidance whenever the urn allows it."""
    R = np.full((6, 5, 5), -1, dtype=int)
    urn = np.tile(np.array([2, 1, 2]), (6, 5, 1))
    last = [-1] * 5
    for (a, s, v) in FIRING:
        c = urn[a, s].astype(float)
        w = c.copy()
        if last[s] >= 0:
            w[last[s]] *= (1 - lam)
        if w.sum() <= 0:
            w = c
        x = rng.choice(3, p=w / w.sum())
        R[a, s, v] = x
        urn[a, s, x] -= 1
        last[s] = x
    return R


class Library:
    """LIB: K fixed full-round schemes, each a uniform 6 x 5 array of orderings,
    drawn with replacement round after round."""
    def __init__(self, rng, K):
        self.schemes = [gen_null(rng) for _ in range(K)]
        self.K = K

    def draw(self, rng):
        return self.schemes[rng.integers(0, self.K)].copy()


# ---------------------------------------------------------------- the models
# Each model object exposes
#   predict(state, a, s, v) -> length-3 array P(next label)
#   update(state, a, s, v, x)    (state is shared; models read masks from it)
# where state carries the per-row consistency masks and the firing-order urns.

class State:
    def __init__(self):
        self.mask = np.ones((6, 5, N_ORD), dtype=bool)
        self.obs = np.full((6, 5, 5), -1, dtype=int)
        self.urn = np.tile(np.array([2, 1, 2]), (6, 5, 1))
        self.last = [-1] * 5
        self.n_obs = 0

    def update(self, a, s, v, x):
        self.mask[a, s] &= POS_LAB[v, x]
        self.obs[a, s, v] = x
        self.urn[a, s, x] -= 1
        self.last[s] = x
        self.n_obs += 1


def _row_weight(mask, lam, restrict):
    """lam * |mask ∩ restrict|/|restrict| + (1-lam) |mask|/30 for a row mask."""
    return lam * (mask & restrict).sum() / restrict.sum() + (1 - lam) * mask.sum() / N_ORD


class M0:
    name = "M0 independent uniform"
    lam = None

    def predict(self, st, a, s, v):
        m = st.mask[a, s]
        return np.array([(m & POS_LAB[v, x]).sum() for x in range(3)], float) / max(m.sum(), 1)


class A4:
    name = "A4 no adjacent repeat"

    def __init__(self, lam):
        self.lam = lam

    def predict(self, st, a, s, v):
        m = st.mask[a, s]
        w = np.array([_row_weight(m & POS_LAB[v, x], self.lam, CLEAN) for x in range(3)])
        tot = w.sum()
        return w / tot if tot > 0 else np.full(3, 1 / 3)


class A1:
    name = "A1 shared ordering per station"

    def __init__(self, lam):
        self.lam = lam

    def predict(self, st, a, s, v):
        lam = self.lam
        others = np.ones(N_ORD)
        for b in range(6):
            if b != a:
                m = st.mask[b, s]
                others *= lam * m + (1 - lam) * m.sum() / N_ORD
        w = np.empty(3)
        for x in range(3):
            m = st.mask[a, s] & POS_LAB[v, x]
            w[x] = (others * (lam * m + (1 - lam) * m.sum() / N_ORD)).mean()
        tot = w.sum()
        return w / tot if tot > 0 else np.full(3, 1 / 3)


class A2o:
    name = "A2o master orbit per station"

    def __init__(self, lam):
        self.lam = lam

    def predict(self, st, a, s, v):
        lam = self.lam
        others = np.ones(6)
        for b in range(6):
            if b != a:
                m = st.mask[b, s]
                others *= lam * (ORBIT & m).sum(1) / 5 + (1 - lam) * m.sum() / N_ORD
        w = np.empty(3)
        for x in range(3):
            m = st.mask[a, s] & POS_LAB[v, x]
            w[x] = (others * (lam * (ORBIT & m).sum(1) / 5 + (1 - lam) * m.sum() / N_ORD)).mean()
        tot = w.sum()
        return w / tot if tot > 0 else np.full(3, 1 / 3)


class A5:
    name = "A5 cross-station coupling"

    def __init__(self, lam):
        self.lam = lam

    def predict(self, st, a, s, v):
        lam = self.lam
        others = np.ones(N_ORD)
        for t in range(5):
            if t != s:
                m = st.mask[a, t]
                others *= lam * m + (1 - lam) * m.sum() / N_ORD
        w = np.empty(3)
        for x in range(3):
            m = st.mask[a, s] & POS_LAB[v, x]
            w[x] = (others * (lam * m + (1 - lam) * m.sum() / N_ORD)).mean()
        tot = w.sum()
        return w / tot if tot > 0 else np.full(3, 1 / 3)


class A7:
    name = "A7 firing-order repeat avoidance"

    def __init__(self, lam):
        self.lam = lam

    def predict(self, st, a, s, v):
        c = st.urn[a, s].astype(float)
        w = c.copy()
        if st.last[s] >= 0:
            w[st.last[s]] *= (1 - self.lam)
        if w.sum() <= 0:
            w = c
        return w / w.sum()


class LIB:
    """Scheme library with K entries, drawn with replacement.  `known` holds the
    distinct full rounds seen so far (from earlier rounds)."""
    name = "LIB scheme reuse across rounds"

    def __init__(self, K):
        self.K = K
        self.lam = None
        self.known = []                   # list of 6x5x5 int arrays
        self.m0 = M0()

    def _p_prefix_known(self, st):
        # P(prefix) contributed by known schemes: count of known schemes consistent with obs
        obs = st.obs
        seen = obs >= 0
        n = 0
        for sch in self.known:
            if np.all(sch[seen] == obs[seen]):
                n += 1
        return n

    def predict(self, st, a, s, v):
        d = len(self.known)
        p0 = self.m0.predict(st, a, s, v)
        if d == 0:
            return p0
        obs = st.obs
        seen = obs >= 0
        cons = [sch for sch in self.known if np.all(sch[seen] == obs[seen])]
        # P(prefix) under LIB = (#consistent known)/K + (K-d)/K * P0(prefix); P0(prefix)=prod |mask|/30
        logp0 = np.sum(np.log(st.mask.sum(2) / N_ORD))
        p0_prefix = np.exp(logp0)
        w = np.zeros(3)
        for x in range(3):
            n_x = sum(1 for sch in cons if sch[a, s, v] == x)
            w[x] = n_x / self.K + (self.K - d) / self.K * p0_prefix * p0[x]
        tot = w.sum()
        return w / tot if tot > 0 else p0

    def end_of_round(self, R):
        if not any(np.array_equal(R, k) for k in self.known):
            self.known.append(R.copy())


LAM_GRID = (0.25, 0.5, 0.75, 1.0)


def build_library(K_lib=20, lib_model=None):
    """The model library with prior weights.  Prior: 1/2 on M0, the rest split
    equally over the alternative families, uniform over lam inside a family."""
    fams = {
        "A1": [A1(l) for l in LAM_GRID],
        "A2o": [A2o(l) for l in LAM_GRID],
        "A4": [A4(l) for l in LAM_GRID],
        "A5": [A5(l) for l in LAM_GRID],
        "A7": [A7(l) for l in LAM_GRID],
    }
    if lib_model is not None:
        fams["LIB"] = [lib_model]
    models, prior, fam_of = [M0()], [0.5], ["M0"]
    n_f = len(fams)
    for f, ms in fams.items():
        for m in ms:
            models.append(m)
            prior.append(0.5 / n_f / len(ms))
            fam_of.append(f)
    return models, np.array(prior), fam_of


# ---------------------------------------------------------------- inference
def run_round(R, models, prior, fam_of, oracle_idx=None):
    """Process one round in firing order.  Returns per-shot log-loss and hit
    indicators for the counting baseline (M0), the posterior mixture, and an
    optional oracle, plus the final log-likelihood vector."""
    st = State()
    n_m = len(models)
    loglik = np.zeros(n_m)
    ll_count, ll_mix, ll_or = [], [], []
    hit_count, hit_mix, hit_or = [], [], []
    logprior = np.log(prior)
    for (a, s, v) in FIRING:
        x = R[a, s, v]
        P = np.array([m.predict(st, a, s, v) for m in models])      # n_m x 3
        lw = logprior + loglik
        lw -= lw.max()
        w = np.exp(lw)
        w /= w.sum()
        pmix = w @ P
        pmix = pmix / pmix.sum()
        p0 = P[0]
        ll_count.append(-np.log(max(p0[x], 1e-300)))
        ll_mix.append(-np.log(max(pmix[x], 1e-300)))
        hit_count.append(int(np.argmax(p0) == x))
        hit_mix.append(int(np.argmax(pmix) == x))
        if oracle_idx is not None:
            po = P[oracle_idx]
            ll_or.append(-np.log(max(po[x], 1e-300)))
            hit_or.append(int(np.argmax(po) == x))
        loglik += np.log(np.maximum(P[:, x], 1e-300))
        st.update(a, s, v, x)
    return dict(ll_count=np.array(ll_count), ll_mix=np.array(ll_mix), ll_or=np.array(ll_or),
                hit_count=np.array(hit_count), hit_mix=np.array(hit_mix), hit_or=np.array(hit_or),
                loglik=loglik)


def family_posterior(loglik, prior, fam_of):
    lw = np.log(prior) + loglik
    lw -= lw.max()
    w = np.exp(lw)
    w /= w.sum()
    fams = sorted(set(fam_of), key=lambda f: (f != "M0", f))
    post = {f: w[[i for i, g in enumerate(fam_of) if g == f]].sum() for f in fams}
    # log Bayes factor of each family vs M0 (prior-free inside the family: uniform over lam)
    lbf = {}
    for f in fams:
        idx = [i for i, g in enumerate(fam_of) if g == f]
        lbf[f] = (np.logaddexp.reduce(loglik[idx]) - np.log(len(idx))) - loglik[0]
    return post, lbf


# ---------------------------------------------------------------- demo
def _block_means(arr, blocks=5):
    return arr.reshape(blocks, -1).mean(1)


def _to_obj(R):
    Rt = np.empty((6, 5, 5), dtype=object)
    for a in range(6):
        for s in range(5):
            Rt[a, s] = np.array([LABELS[i] for i in R[a, s]], dtype=object)
    return Rt


def demo(reps=60, seed=3, K_lib=20, scen=None, do_single=True, do_lib=True):
    rng = np.random.default_rng(seed)
    models, prior, fam_of = build_library()
    fams = sorted(set(fam_of), key=lambda f: (f != "M0", f))
    # the original min-p battery, for side-by-side comparison
    try:
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import single_round_independence_test as T
        tab = T.build_null_table(4000, seed=5)
        mn = T._minp_null(tab)
        have_T = True
    except Exception:
        have_T = False

    scenarios = [
        ("H0 independent", lambda r: gen_null(r), None, "M0"),
        ("A1 shared lam=1", lambda r: gen_shared(r, 1.0), 1.0, "A1"),
        ("A1 shared lam=.5", lambda r: gen_shared(r, 0.5), 0.5, "A1"),
        ("A2 master+offset lam=1", lambda r: gen_master_offset(r, 1.0), 1.0, "A2o"),
        ("A2 master+offset lam=.5", lambda r: gen_master_offset(r, 0.5), 0.5, "A2o"),
        ("A2 master+offset lam=.25", lambda r: gen_master_offset(r, 0.25), 0.25, "A2o"),
        ("A4 no-adjacent lam=1", lambda r: gen_no_adjacent(r, 1.0), 1.0, "A4"),
        ("A4 no-adjacent lam=.5", lambda r: gen_no_adjacent(r, 0.5), 0.5, "A4"),
        ("A5 cross-station lam=1", lambda r: gen_cross_station(r, 1.0), 1.0, "A5"),
        ("A5 cross-station lam=.5", lambda r: gen_cross_station(r, 0.5), 0.5, "A5"),
        ("A7 firing-order lam=1", lambda r: gen_firing_order_avoid(r, 1.0), 1.0, "A7"),
        ("A7 firing-order lam=.5", lambda r: gen_firing_order_avoid(r, 0.5), 0.5, "A7"),
    ]

    if scen is not None:
        scenarios = [scenarios[i] for i in scen]
    acc_rows = []
    if not do_single:
        scenarios = []
    if scenarios:
        print(f"=== SINGLE ROUND: posterior family mass after 150 shots, {reps} rounds per scenario ===")
        print(f"{'truth':<26}" + "".join(f"{f:>7}" for f in fams) + f"{'lnBF(true)':>12}{'min-p rej':>11}")
    for label, gen, lam, true_fam in scenarios:
        post_acc = {f: 0.0 for f in fams}
        lbf_true = []
        rejT = 0
        acc_c, acc_m, acc_o = np.zeros(5), np.zeros(5), np.zeros(5)
        ll_c, ll_m, ll_o = 0.0, 0.0, 0.0
        # oracle index: the model in the library with the true family and lam (nearest on grid)
        oracle_idx = None
        if true_fam != "M0":
            cands = [i for i, (m, f) in enumerate(zip(models, fam_of)) if f == true_fam]
            oracle_idx = min(cands, key=lambda i: abs(models[i].lam - lam))
        else:
            oracle_idx = 0
        for _ in range(reps):
            R = gen(rng)
            if have_T:
                rejT += T.test_round(_to_obj(R), tab, mn)[3] <= 0.05
            out = run_round(R, models, prior, fam_of, oracle_idx)
            post, lbf = family_posterior(out["loglik"], prior, fam_of)
            for f in fams:
                post_acc[f] += post[f] / reps
            lbf_true.append(lbf[true_fam] if true_fam != "M0" else -max(lbf[f] for f in fams if f != "M0"))
            acc_c += _block_means(out["hit_count"]) / reps
            acc_m += _block_means(out["hit_mix"]) / reps
            acc_o += _block_means(out["hit_or"]) / reps
            ll_c += out["ll_count"].mean() / reps
            ll_m += out["ll_mix"].mean() / reps
            ll_o += out["ll_or"].mean() / reps
        med = np.median(lbf_true)
        print(f"{label:<26}" + "".join(f"{post_acc[f]:>7.2f}" for f in fams) + f"{med:>12.1f}"
              + (f"{rejT/reps:>11.2f}" if have_T else ""))
        acc_rows.append((label, acc_c, acc_m, acc_o, ll_c, ll_m, ll_o))

    if scenarios:
        print("\nlnBF(true): median natural-log Bayes factor of the true family against M0 "
              "(H0 row: minus the largest alternative lnBF, positive favours M0). "
              "min-p rej: rejection rate of the original single-round battery at alpha 0.05.")
        print(f"\n=== PREQUENTIAL PREDICTION: accuracy by 30-shot block and mean log-loss per shot ===")
        print("counting = M0 predictor (the 60.7 per cent ceiling); mixture = posterior predictive "
              "over the library; oracle = true model, true lam")
        print(f"{'truth':<26}{'predictor':<10}" + "".join(f"{'b'+str(b+1):>7}" for b in range(5)) + f"{'all':>7}{'logloss':>9}")
    for label, acc_c, acc_m, acc_o, ll_c, ll_m, ll_o in acc_rows:
        for nm, acc, ll in (("counting", acc_c, ll_c), ("mixture", acc_m, ll_m), ("oracle", acc_o, ll_o)):
            print(f"{label:<26}{nm:<10}" + "".join(f"{v:>7.3f}" for v in acc) + f"{acc.mean():>7.3f}{ll:>9.4f}")
        print()

    # ---------------- cross-round scheme reuse
    if not do_lib:
        return
    n_rep = max(10, reps // 3)
    print(f"=== SCHEME REUSE ACROSS ROUNDS: library of K={K_lib} schemes, 12 rounds, {n_rep} replications ===")
    print("Within a round a library scheme is exactly H0-distributed, so the single-round battery is blind to it.")
    print("Across rounds an exact repeat of a 150-target round has lnBF = 30 ln 30 - ln K against independence.")
    n_rounds = 12
    acc_rounds_mix = np.zeros(n_rounds)
    acc_rounds_cnt = np.zeros(n_rounds)
    lbf_rounds = np.zeros(n_rounds)
    rej_T = []
    for _ in range(n_rep):
        lib = Library(rng, K_lib)
        lib_model = LIB(K_lib)
        models_l, prior_l, fam_l = build_library(lib_model=lib_model)
        d = 0
        cum_lbf = 0.0
        for t in range(n_rounds):
            R = lib.draw(rng)
            out = run_round(R, models_l, prior_l, fam_l)
            acc_rounds_mix[t] += out["hit_mix"].mean() / n_rep
            acc_rounds_cnt[t] += out["hit_count"].mean() / n_rep
            # cross-round lnBF for LIB(K) vs M0, computed from round-level match events
            match = any(np.array_equal(R, k) for k in lib_model.known)
            cum_lbf += (30 * np.log(30) - np.log(K_lib)) if match else np.log((K_lib - d) / K_lib)
            lbf_rounds[t] += cum_lbf / n_rep
            if have_T:
                rej_T.append(T.test_round(_to_obj(R), tab, mn)[3] <= 0.05)
            lib_model.end_of_round(R)
            d = len(lib_model.known)
    print(f"{'round':<8}" + "".join(f"{t+1:>7}" for t in range(n_rounds)))
    print(f"{'count.':<8}" + "".join(f"{v:>7.3f}" for v in acc_rounds_cnt))
    print(f"{'mixture':<8}" + "".join(f"{v:>7.3f}" for v in acc_rounds_mix))
    print(f"{'cum lnBF':<8}" + "".join(f"{v:>7.0f}" for v in lbf_rounds))
    if have_T:
        print(f"single-round min-p battery rejection rate on library rounds: {np.mean(rej_T):.3f} (alpha 0.05)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--scen", type=str, default=None, help="comma-separated scenario indices")
    ap.add_argument("--no-single", action="store_true")
    ap.add_argument("--no-lib", action="store_true")
    args = ap.parse_args()
    if args.demo:
        scen = [int(i) for i in args.scen.split(",")] if args.scen else None
        demo(reps=args.reps, seed=args.seed, scen=scen,
             do_single=not args.no_single, do_lib=not args.no_lib)
    else:
        ap.print_help()
