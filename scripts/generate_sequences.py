#!/usr/bin/env python3
"""Generation launcher for mROSE sequence generation.

By default this script prints ready-to-run commands and checks whether local
checkpoints are available. Use --run to execute one or more generation examples.
"""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    "5utr": {
        "label": "5' UTR",
        "script": ROOT / "generation" / "5utr" / "generate_5utr.py",
        "checkpoint": ROOT / "generation" / "5utr" / "Model.pth",
        "input": ROOT / "generation" / "examples" / "5utr_template.fasta",
        "output_dir": ROOT / "outputs" / "generation" / "5utr_example",
        "args": [
            "--num_samples", "10000",
            "--top_k", "5",
            "--device", "cuda:0",
            "--output_prefix", "example_5utr",
        ],
    },
    "cds": {
        "label": "CDS",
        "script": ROOT / "generation" / "cds" / "generate_cds.py",
        "checkpoint": ROOT / "generation" / "cds" / "Model.pth",
        "input": ROOT / "generation" / "examples" / "cds_template.fasta",
        "output_dir": ROOT / "outputs" / "generation" / "cds_example",
        "args": [
            "--num_samples", "10000",
            "--top_k", "5",
            "--device", "cuda:0",
            "--mfe_weight", "0",
        ],
    },
    "3utr": {
        "label": "3' UTR",
        "script": ROOT / "generation" / "3utr" / "generate_3utr.py",
        "checkpoint": ROOT / "generation" / "3utr" / "Model.pth",
        "input": ROOT / "generation" / "examples" / "3utr_template.fasta",
        "output_dir": ROOT / "outputs" / "generation" / "3utr_example",
        "args": [
            "--num_samples", "10000",
            "--top_k", "5",
            "--device", "cuda:0",
            "--output_prefix", "example_3utr",
        ],
    },
    "full_length": {
        "label": "Full-length mRNA",
        "script": ROOT / "generation" / "full_length" / "generate_full_length.py",
        "checkpoint": [
            ROOT / "generation" / "5utr" / "Model.pth",
            ROOT / "generation" / "cds" / "Model.pth",
            ROOT / "generation" / "3utr" / "Model.pth",
        ],
        "full_mrna_input": ROOT / "generation" / "examples" / "full_mrna_template.fasta",
        "output_dir": ROOT / "outputs" / "generation" / "full_length_example",
        "args": [
            "--num_samples", "100",
            "--top_k", "5",
            "--device", "cuda:0",
            "--output_prefix", "example_full_length",
        ],
    },
}

REQUIRED_MODULES = ["numpy", "pandas", "torch", "Bio", "sklearn", "scipy", "tqdm"]
VIENNA_TASKS = {"5utr", "3utr"}


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def command_for(task: str, python_bin: str) -> list[str]:
    cfg = TASKS[task]
    if task == "full_length":
        return [
            python_bin,
            str(cfg["script"]),
            "--full_mrna_fasta",
            str(cfg["full_mrna_input"]),
            "--output_dir",
            str(cfg["output_dir"]),
            *cfg["args"],
        ]
    return [
        python_bin,
        str(cfg["script"]),
        "--checkpoint",
        str(cfg["checkpoint"]),
        "--input_fasta",
        str(cfg["input"]),
        "--output_dir",
        str(cfg["output_dir"]),
        *cfg["args"],
    ]


def print_status(python_bin: str) -> None:
    print("mROSE generation example")

    print("Dependency check:")
    for name in REQUIRED_MODULES:
        print(f"  {name:10s} {'OK' if module_available(name) else 'missing'}")
    vienna_ok = module_available("ViennaRNA") or module_available("RNA")
    print(f"  {'ViennaRNA':10s} {'OK' if vienna_ok else 'missing'}")
    print()

    print("Checkpoint check:")
    for key, cfg in TASKS.items():
        checkpoints = cfg["checkpoint"] if isinstance(cfg["checkpoint"], list) else [cfg["checkpoint"]]
        status = "OK" if all(checkpoint.exists() for checkpoint in checkpoints) else "missing"
        print(f"  {key:4s} {status}")
    print()

    print("Commands:")
    for key in TASKS:
        cmd = command_for(key, python_bin)
        print(f"\n# {TASKS[key]['label']}")
        print(shlex.join(cmd))


def validate_before_run(task: str) -> list[str]:
    problems: list[str] = []
    cfg = TASKS[task]
    checkpoints = cfg["checkpoint"] if isinstance(cfg["checkpoint"], list) else [cfg["checkpoint"]]
    for checkpoint in checkpoints:
        if not checkpoint.exists():
            problems.append(f"missing checkpoint: {checkpoint}")
    for name in REQUIRED_MODULES:
        if not module_available(name):
            problems.append(f"missing Python module: {name}")
    if (task in VIENNA_TASKS or task == "full_length") and not (module_available("ViennaRNA") or module_available("RNA")):
        problems.append("missing Python module: ViennaRNA or RNA")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Print or run mROSE generation examples.")
    parser.add_argument(
        "--run",
        choices=["none", "5utr", "cds", "3utr", "full_length", "all"],
        default="none",
        help="Run an example task. Default only prints commands and checks.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use.")
    args = parser.parse_args()

    print_status(args.python)
    if args.run == "none":
        return 0

    tasks = list(TASKS) if args.run == "all" else [args.run]
    for task in tasks:
        problems = validate_before_run(task)
        if problems:
            print(f"\nCannot run {task} example:")
            for problem in problems:
                print(f"  - {problem}")
            return 2

        TASKS[task]["output_dir"].mkdir(parents=True, exist_ok=True)
        cmd = command_for(task, args.python)
        print(f"\nRunning {TASKS[task]['label']} example...")
        subprocess.run(cmd, cwd=ROOT, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
