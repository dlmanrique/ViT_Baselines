#!/usr/bin/env bash
# Frame-wise ViT triplet recognition on the official CholecT45 5-fold
# cross-validation: one training run of main.py per test fold (1..5), one
# after the other. Stops at the first fold that fails. The 5 runs share one
# wandb group, triplet_cv_<launch date and time>, so wandb aggregates them.
#
# Usage (from any folder):
#   bash run_triplet_cv.sh                      # 1 epoch per fold (pipeline test)
#   bash run_triplet_cv.sh 15                   # 15 epochs per fold
#   bash run_triplet_cv.sh 15 --lr 1e-4         # extra arguments go to main.py
#   bash run_triplet_cv.sh 1 --device cuda:1    # later arguments override the defaults below
#
# PYTHON selects the interpreter (default: the Base_Leo environment).
set -euo pipefail

EPOCHS=${1:-1}
if [ $# -gt 0 ]; then shift; fi
PYTHON=${PYTHON:-/home/oem/anaconda3/envs/Base_Leo/bin/python}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP="triplet_cv_$(date +%Y-%m-%d_%H-%M-%S)"

for fold in 1 2 3 4 5; do
    echo "===== CholecT45 fold ${fold}/5 | ${EPOCHS} epoch(s) | wandb group ${GROUP} ====="
    "${PYTHON}" "${SCRIPT_DIR}/main.py" --task triplet_recognition --fold "${fold}" --epochs "${EPOCHS}" \
        --device cuda:0 --wandb_group "${GROUP}" "$@"
done
