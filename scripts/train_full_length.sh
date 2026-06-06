#!/usr/bin/env bash
set -euo pipefail

python -m mrose.full_length_fusion \
  --data_base_dir experiments/full_length_stability/example \
  --model_base_dir outputs/full_length_stability_example
