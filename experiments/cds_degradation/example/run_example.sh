#!/usr/bin/env bash
set -euo pipefail

# The CDS training script currently expects ./Data/train.csv and ./Data/test.csv
# relative to the current working directory and uses torchrun/DDP.
cd experiments/cds_degradation/example
torchrun --nproc_per_node=1 ../train_cds.py   --epochs 1   --batch_size 4   --num_workers 0   --num_trials 1   --output_dir ../../outputs/cds_degradation_example
