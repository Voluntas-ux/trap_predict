import os, sys
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")]
import numpy as np, time
import single_round_independence_test as T
tab = T.build_null_table(20000, seed=101); mn = T._minp_null(tab)
rng = np.random.default_rng(202)
N=20000; gps=np.empty(N); t=time.time()
for i in range(N):
    gps[i]=T.test_round(T.gen_null(rng), tab, mn)[3]
print('elapsed', round(time.time()-t,1))
for a in (0.01,0.05,0.10):
    print(f'alpha={a}: size={np.mean(gps<=a):.4f}  (SE {np.sqrt(a*(1-a)/N):.4f})')
# exchangeable check: the null table's own minp distribution vs test rounds
