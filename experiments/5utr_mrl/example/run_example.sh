#!/usr/bin/env bash
set -euo pipefail

# From repository root:
python experiments/5utr_mrl/train_5utr_mrl.py   --train_file experiments/5utr_mrl/example/train.csv   --test_file experiments/5utr_mrl/example/test.csv   --epochs 1   --batch_size 4   --device cpu   --output_dir outputs/5utr_mrl_example
