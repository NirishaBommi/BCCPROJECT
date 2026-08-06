#!/usr/bin/env bash
# Usage: bash scripts/run_predict.sh <image> <checkpoint>
IMAGE=${1:?"Provide image path"}
CKPT=${2:?"Provide checkpoint path"}
python src/predict.py --image "$IMAGE" --checkpoint "$CKPT" \
    --num_classes 2 --temperature 1.31 --save_dir gradcam_outputs
