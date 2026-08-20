"""Independent verification of the null moments and structural claims in
single_round_test_note.md. Closed forms where they exist, large vectorised
Monte Carlo where they do not."""
import os, sys
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")]
import numpy as np
from itertools import permutations
from math import comb

rng = np.random.default_rng(123)
LAB = {"L": 0, "S": 1, "R": 2}
ORD = sorted(set(permutations("LLSRR")))
O = np.array([[LAB[c] for c in o] for o in ORD])          # 30 x 5 ints
assert O.shape == (30, 5)
p = np.array([0.4, 0.2, 0.4])

# ---------- closed forms -----------------------------------------------
print("=== closed-form null moments ===")
# S1: 5 stations x C(6,2) pairs, P(identical)=1/30, pairwise covariances are zero
m1 = 5 * comb(6, 2) / 30
v1 = 5 * comb(6, 2) * (1 / 30) * (29 / 30)
print(f"S1 mean {m1:.4f}  sd {v1**0.5:.4f}")
# S2: occupied boxes, 6 balls in 6 boxes, per station
m, n = 6, 6
mean_occ = m * (1 - (1 - 1 / m) ** n)
var_occ = m * (1 - 1 / m) ** n + m * (m - 1) * (1 - 2 / m) ** n - m**2 * (1 - 1 / m) ** (2 * n)
print(f"S2 mean {5*mean_occ:.4f}  sd {(5*var_occ)**0.5:.4f}")
# S3 / S6: Pearson X2 for Multinomial(6,p); exact mean k-1=2,
# exact var = 2(k-1) + (sum 1/p - k^2 - 2k + 2)/n
k, nn = 3, 6
v_col = 2 * (k - 1) + (np.sum(1 / p) - k**2 - 2 * k + 2) / nn
print(f"per-column X2: mean 2, var {v_col:.4f}, sd {v_col**0.5:.4f}")
print(f"S6 (5 independent columns): mean 10, sd {(5*v_col)**0.5:.4f}")
print(f"S3 IF the 25 columns were independent: mean 50, sd {(25*v_col)**0.5:.4f}")
# S4: adjacent repeats in a uniform ordering of LLSRR, exact distribution
adj = (O[:, :-1] == O[:, 1:]).sum(1)
m4, v4 = adj.mean(), adj.var()
print(f"S4 per-row mean {m4:.4f} var {v4:.4f}; 30 rows: mean {30*m4:.4f} sd {(30*v4)**0.5:.4f}")
# S5: 6 athletes x C(5,2) pairs
m5 = 6 * comb(5, 2) / 30
v5 = 6 * comb(5, 2) * (1 / 30) * (29 / 30)
print(f"S5 mean {m5:.4f}  sd {v5**0.5:.4f}")

# ---------- vectorised MC for S3 and the column-dependence question -------
print("\n=== vectorised MC, 2,000,000 null stations ===")
N = 2_000_000
idx = rng.integers(0, 30, size=(N, 6))
rows = O[idx]                                             # N x 6 x 5
exp = 6 * p
X2 = np.zeros((N, 5))
for v in range(5):
    col = rows[:, :, v]
    counts = np.stack([(col == c).sum(1) for c in range(3)], axis=1)   # N x 3
    X2[:, v] = (((counts - exp) ** 2) / exp).sum(1)
station_sum = X2.sum(1)
print(f"per-column X2 mean {X2.mean():.4f} var {X2.var():.4f}")
C = np.cov(X2.T)
print("covariance matrix of the five visit-columns at one station:")
print(np.round(C, 4))
offdiag = C[~np.eye(5, dtype=bool)].mean()
print(f"mean off-diagonal covariance {offdiag:.4f}  (zero if columns were independent)")
print(f"per-station sum: mean {station_sum.mean():.4f} var {station_sum.var():.4f}")
print(f"S3 = 5 stations: mean {5*station_sum.mean():.4f} sd {(5*station_sum.var())**0.5:.4f}")
print(f"  vs independent-column prediction sd {(25*v_col)**0.5:.4f}")

# ---------- the no-adjacent-repeat orbit structure --------------------------
print("\n=== A4 support and its orbit occupancy ===")
def shift(o, k):
    return tuple(o[(i + k) % 5] for i in range(5))
orbits, seen = [], set()
for o in ORD:
    if o in seen:
        continue
    orb = {shift(o, k) for k in range(5)}
    orbits.append(orb); seen |= orb
orbit_of = {o: i for i, orb in enumerate(orbits) for o in orb}
clean = [o for o in ORD if all(o[i] != o[i + 1] for i in range(4))]
print("clean orderings:", len(clean))
occ = np.bincount([orbit_of[o] for o in clean], minlength=6)
print("clean orderings per orbit:", occ)
q = occ / occ.sum()
exp_distinct = np.sum(1 - (1 - q) ** 6)
print(f"expected distinct orbits per station under A4(lam=1): {exp_distinct:.3f} "
      f"(null {mean_occ:.3f}); S2 expectation over 5 stations {5*exp_distinct:.2f}")

# ---------- exact covariance between two visit-columns at one station -------
print("\n=== exact Cov(X2_v, X2_v') at one station, by enumeration ===")
from fractions import Fraction as F
from math import factorial
cells = {}
for o in ORD:
    k = (o[0], o[1]); cells[k] = cells.get(k, 0) + F(1, 30)
cells = {k: v for k, v in cells.items() if v > 0}
keys = list(cells)
pf = {"L": F(2, 5), "S": F(1, 5), "R": F(2, 5)}
def x2f(counts):
    return sum((F(counts.get(c, 0)) - 6 * pf[c]) ** 2 / (6 * pf[c]) for c in "LSR")
def comps(n, k):
    if k == 1:
        yield (n,); return
    for i in range(n + 1):
        for rest in comps(n - i, k - 1):
            yield (i,) + rest
E1 = E12 = F(0)
for comp in comps(6, len(keys)):
    prob = F(factorial(6))
    c1, c2 = {}, {}
    for c, k in zip(comp, keys):
        prob *= cells[k] ** c / factorial(c)
        c1[k[0]] = c1.get(k[0], 0) + c; c2[k[1]] = c2.get(k[1], 0) + c
    a, b = x2f(c1), x2f(c2)
    E1 += prob * a; E12 += prob * a * b
cov = E12 - E1 * E1
var_s3 = 5 * (5 * F(7, 2) + 20 * cov)
print(f"Cov = {cov} ; exact Var(S3) = {var_s3} ; sd = {float(var_s3)**0.5:.4f}")
