#!/usr/bin/env bash
set -e
source .venv/bin/activate
python train.py --dataset-dir data/brain_tumor_dataset --epochs 12 --batch-size 32