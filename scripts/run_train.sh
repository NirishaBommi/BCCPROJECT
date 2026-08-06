#!/usr/bin/env bash
# Usage: bash scripts/run_train.sh [binary|subtype]
MODE=${1:-binary}
CLASSES=$([ "$MODE" = "subtype" ] && echo 5 || echo 2)
python src/train.py --train_csv data/train.csv --val_csv data/val.csv \
    --test_csv data/test.csv --root_dir data --mode $MODE \
    --num_classes $CLASSES --batch_size 32 --epochs 120 \
    --lr 1e-4 --output_dir outputs/$MODE
