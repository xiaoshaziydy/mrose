from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mrose.full_length_fusion import run_multi_folder_grid_search

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_base_dir", type=str, default="experiments/full_length_stability/example")
    parser.add_argument("--model_base_dir", type=str, default="outputs/full_length_stability_example")
    args = parser.parse_args()
    run_multi_folder_grid_search(args.data_base_dir, args.model_base_dir)
