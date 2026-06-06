#!/usr/bin/env bash
set -euo pipefail

# The 3'UTR RBP script expects fold1/fold2/fold3/fold4 and test.fa
# relative to the current working directory and uses torchrun/DDP.
cd experiments/3utr_rbp/example/AKAP1_HepG2
torchrun --nproc_per_node=1 ../../train_3utr_rbp.py   --epochs 1   --batch_size 4   --output_dir ../../../outputs/3utr_rbp_example
