#!/usr/bin/env bash
# Triplet recognition on the official CholecT45 5-fold cross-validation with
# main.py: one run per fold, one after the other, all in one wandb group
# triplet_<stage>_cv_<launch date and time>. Stops at the first fold that fails.
#   vit               step 1: frame-wise ViT of each test fold (--fold 1..5)
#   tcm, mrtt, frame  step 2: temporal module on the frozen ViT of each fold (VIT_RUNS)
#
# Usage (from any folder):
#   bash run_triplet_cv.sh vit                                # ViT, main.py defaults (15 epochs)
#   bash run_triplet_cv.sh vit --epochs 5 --wandb_mode disabled
#   bash run_triplet_cv.sh tcm
#   bash run_triplet_cv.sh mrtt --ks 4                        # extra arguments go to main.py
#
# PYTHON selects the interpreter (default: the Base_Leo environment).
set -euo pipefail

STAGE=${1:?usage: run_triplet_cv.sh <vit|tcm|mrtt|frame> [main.py arguments]}
shift
PYTHON=${PYTHON:-/home/oem/anaconda3/envs/Base_Leo/bin/python}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP="triplet_${STAGE}_cv_$(date +%Y-%m-%d_%H-%M-%S)"

# Frozen ViT of each fold for step 2: main.py, 15 epochs, lr 5e-5 (launched on 2026-10-19)
VIT_RUNS=(
    2026-10-19_19-13-11_triplet_recognition_fold1
    2026-10-19_19-46-24_triplet_recognition_fold2
    2026-10-19_20-18-17_triplet_recognition_fold3
    2026-10-19_20-51-20_triplet_recognition_fold4
    2026-10-19_21-24-46_triplet_recognition_fold5
)

case "${STAGE}" in
    vit)
        for fold in 1 2 3 4 5; do
            echo "===== CholecT45 fold ${fold}/5 | ViT | wandb group ${GROUP} ====="
            "${PYTHON}" "${SCRIPT_DIR}/main.py" --task triplet_recognition --fold "${fold}" --device cuda:0 \
                --wandb_group "${GROUP}" "$@"
        done
        ;;
    tcm|mrtt|frame)
        for i in "${!VIT_RUNS[@]}"; do
            fold=$((i + 1))  # main.py checks that the checkpoint was trained with this test fold
            echo "===== CholecT45 fold ${fold}/5 | ${STAGE} on ${VIT_RUNS[i]} | wandb group ${GROUP} ====="
            "${PYTHON}" "${SCRIPT_DIR}/main.py" --task triplet_recognition --fold "${fold}" \
                --checkpoint "${SCRIPT_DIR}/outputs/${VIT_RUNS[i]}/best.pt" --temporal_model "${STAGE}" \
                --device cuda:0 --wandb_group "${GROUP}" "$@"
        done
        ;;
    *)
        echo "Unknown stage ${STAGE}: use vit, tcm, mrtt or frame" >&2
        exit 1
        ;;
esac
