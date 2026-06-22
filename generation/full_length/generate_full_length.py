#!/usr/bin/env python3
"""Generate full-length mRNA candidates by composing regional mROSE outputs.

This launcher runs the existing 5' UTR, CDS and 3' UTR generation scripts,
then combines candidates by rank:

    full_rank_i = five_utr_rank_i + cds_rank_i + three_utr_rank_i

The regional scripts remain the source of model inference and scoring. This
file only handles orchestration and full-length output assembly.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]


def default_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def run_command(command: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(str(part) for part in command))
    subprocess.run(command, cwd=cwd, check=True)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Expected file was not created: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "sequence" not in rows[0]:
        raise ValueError(f"CSV file must contain a sequence column: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_fasta(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(
                f">full_length_rank_{row['rank']}|"
                f"5utr={row['five_utr_length']}|"
                f"cds={row['cds_length']}|"
                f"3utr={row['three_utr_length']}|"
                f"full={row['full_length']}\n"
            )
            handle.write(str(row["sequence"]) + "\n")


def top_csv_path(region: str, output_dir: Path, output_prefix: str, top_k: int) -> Path:
    if region == "cds":
        return output_dir / "cds" / f"{output_prefix}_cds_top.csv"
    return output_dir / region / f"{output_prefix}_{region}_top{top_k}.csv"


def build_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    five_output = args.output_dir / "5utr"
    cds_output = args.output_dir / "cds"
    three_output = args.output_dir / "3utr"

    # The 5' UTR scorer returns a scalar for a single sample in some
    # environments, so keep the internal generation request at least 2.
    five_num_samples = max(args.num_samples, 2)

    return {
        "5utr": [
            args.python,
            str(args.five_utr_script),
            "--checkpoint",
            str(args.five_utr_checkpoint),
            "--input_fasta",
            str(args.five_utr_fasta),
            "--output_dir",
            str(five_output),
            "--num_samples",
            str(five_num_samples),
            "--top_k",
            str(args.top_k),
            "--device",
            args.device,
            "--temperature",
            str(args.temperature),
            "--output_prefix",
            f"{args.output_prefix}_5utr",
        ],
        "cds": [
            args.python,
            str(args.cds_script),
            "--checkpoint",
            str(args.cds_checkpoint),
            "--input_fasta",
            str(args.cds_fasta),
            "--output_dir",
            str(cds_output),
            "--num_samples",
            str(args.num_samples),
            "--top_k",
            str(args.top_k),
            "--device",
            args.device,
            "--temperature",
            str(args.temperature),
            "--all_csv",
            f"{args.output_prefix}_cds_all_scored.csv",
            "--top_csv",
            f"{args.output_prefix}_cds_top.csv",
            "--top_fasta",
            f"{args.output_prefix}_cds_top.fasta",
            "--mfe_weight",
            str(args.cds_mfe_weight),
            "--batch_size",
            str(args.cds_batch_size),
        ],
        "3utr": [
            args.python,
            str(args.three_utr_script),
            "--checkpoint",
            str(args.three_utr_checkpoint),
            "--input_fasta",
            str(args.three_utr_fasta),
            "--output_dir",
            str(three_output),
            "--num_samples",
            str(args.num_samples),
            "--top_k",
            str(args.top_k),
            "--device",
            args.device,
            "--temperature",
            str(args.temperature),
            "--output_prefix",
            f"{args.output_prefix}_3utr",
            "--match_input_length",
        ],
    }


def merge_outputs(args: argparse.Namespace) -> list[dict[str, object]]:
    five_rows = read_rows(top_csv_path("5utr", args.output_dir, args.output_prefix, args.top_k))
    cds_rows = read_rows(top_csv_path("cds", args.output_dir, args.output_prefix, args.top_k))
    three_rows = read_rows(top_csv_path("3utr", args.output_dir, args.output_prefix, args.top_k))

    merge_count = min(args.top_k, len(five_rows), len(cds_rows), len(three_rows))
    if merge_count < 1:
        raise ValueError("No full-length candidates could be merged.")

    rows: list[dict[str, object]] = []
    for index in range(merge_count):
        five = five_rows[index]["sequence"].upper().replace("U", "T")
        cds = cds_rows[index]["sequence"].upper().replace("U", "T")
        three = three_rows[index]["sequence"].upper().replace("U", "T")
        full = five + cds + three
        rows.append(
            {
                "rank": index + 1,
                "sequence": full,
                "five_utr_sequence": five,
                "cds_sequence": cds,
                "three_utr_sequence": three,
                "five_utr_length": len(five),
                "cds_length": len(cds),
                "three_utr_length": len(three),
                "full_length": len(full),
                "five_utr_score": five_rows[index].get("final_score", ""),
                "cds_score": cds_rows[index].get("final_score", ""),
                "three_utr_score": three_rows[index].get("final_score", ""),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full-length mRNA candidates by running and merging regional mROSE generators."
    )
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used to run regional generators.")
    parser.add_argument("--device", default="cuda:0", help="Device passed to regional generators.")
    parser.add_argument("--num_samples", type=int, default=100, help="Candidates to sample per region.")
    parser.add_argument("--top_k", type=int, default=10, help="Ranked candidates to keep and merge.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--output_dir", type=Path, default=default_path("outputs", "generation", "full_length_example"))
    parser.add_argument("--output_prefix", default="example_full_length")

    parser.add_argument("--five_utr_fasta", type=Path, default=default_path("generation", "examples", "5utr_template.fasta"))
    parser.add_argument("--cds_fasta", type=Path, default=default_path("generation", "examples", "cds_template.fasta"))
    parser.add_argument("--three_utr_fasta", type=Path, default=default_path("generation", "examples", "3utr_template.fasta"))

    parser.add_argument("--five_utr_checkpoint", type=Path, default=default_path("generation", "5utr", "Model.pth"))
    parser.add_argument("--cds_checkpoint", type=Path, default=default_path("generation", "cds", "Model.pth"))
    parser.add_argument("--three_utr_checkpoint", type=Path, default=default_path("generation", "3utr", "Model.pth"))

    parser.add_argument("--five_utr_script", type=Path, default=default_path("generation", "5utr", "generate_5utr.py"))
    parser.add_argument("--cds_script", type=Path, default=default_path("generation", "cds", "generate_cds.py"))
    parser.add_argument("--three_utr_script", type=Path, default=default_path("generation", "3utr", "generate_3utr.py"))

    parser.add_argument("--cds_mfe_weight", type=float, default=0.0, help="CDS MFE weight passed to CDS ranking.")
    parser.add_argument("--cds_batch_size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for region, command in build_commands(args).items():
        print(f"\n=== Generating {region} candidates ===")
        run_command(command, ROOT)

    rows = merge_outputs(args)
    top_csv = args.output_dir / f"{args.output_prefix}_top{args.top_k}.csv"
    top_fasta = args.output_dir / f"{args.output_prefix}_top{args.top_k}.fasta"
    write_csv(top_csv, rows)
    write_fasta(top_fasta, rows)

    print(f"\nSaved full-length CSV: {top_csv}")
    print(f"Saved full-length FASTA: {top_fasta}")
    print("\nTop full-length candidates:")
    for row in rows:
        print(
            f"{row['rank']}: full={row['full_length']} nt "
            f"(5utr={row['five_utr_length']}, cds={row['cds_length']}, 3utr={row['three_utr_length']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
