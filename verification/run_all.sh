#!/usr/bin/env bash
# Regenerate every log in verification/logs/. Total runtime is roughly 25 minutes.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
SRC=../src

echo "[1/7] exact reproduction of the original design note (~2 min)"
python "$SRC/single_round_independence_test.py" --demo --nnull 8000 --nsim 600 > logs/demo_8000_600.txt

echo "[2/7] independent replication, seed 7 (~10 min)"
python - > logs/demo_20000_3000_seed7.txt <<'PY'
import sys; sys.path.insert(0, "../src")
import single_round_independence_test as T
T.run_power_study(n_null=20000, n_sim=3000, seed=7, alpha=0.05)
PY

echo "[3/7] closed forms and the exact covariance (~1 min)"
python verify_moments.py > logs/verify_moments.txt

echo "[4/7] calibration, 20,000 test rounds (~2 min)"
python size_check.py > logs/size_check.txt

echo "[5/7] pooled power (~10 min)"
python verify_pooling.py 400 > logs/pooling_check.txt

echo "[6/7] identification and prequential prediction (~3 min)"
python "$SRC/trap_controller_inference.py" --demo --reps 80 --seed 3 --no-lib > logs/proto_single.txt

echo "[7/7] cross-round scheme reuse (~2 min)"
python "$SRC/trap_controller_inference.py" --demo --reps 60 --seed 3 --no-single > logs/proto_lib.txt

echo "done. logs written to verification/logs/"
