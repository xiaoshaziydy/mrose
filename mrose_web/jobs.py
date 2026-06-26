"""Background job helpers for the mROSE FastAPI service."""

from __future__ import annotations

import json
import os
import re
import csv
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
JOB_ROOT = Path(os.environ.get("MROSE_WEB_JOB_DIR", ROOT / "outputs" / "web_jobs"))
PYTHON = os.environ.get("MROSE_WEB_PYTHON", sys.executable)
MAX_WORKERS = int(os.environ.get("MROSE_WEB_MAX_WORKERS", "1"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MROSE_WEB_MAX_SEQUENCE_LENGTH", "6000"))
MAX_SAMPLES = int(os.environ.get("MROSE_WEB_MAX_SAMPLES", "10000"))
MAX_TOP_K = int(os.environ.get("MROSE_WEB_MAX_TOP_K", "100"))
DEFAULT_DEVICE = os.environ.get("MROSE_WEB_DEFAULT_DEVICE", "cpu")

EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)
VALID_SEQUENCE = re.compile(r"^[ACGTUacgtu\s]+$")

REGIONS = {
    "5utr": {
        "label": "5' UTR",
        "script": ROOT / "generation" / "5utr" / "generate_5utr.py",
        "checkpoint": ROOT / "generation" / "5utr" / "Model.pth",
        "prefix": "mrose_5utr",
        "extra_args": ["--output_prefix", "mrose_5utr"],
    },
    "cds": {
        "label": "CDS",
        "script": ROOT / "generation" / "cds" / "generate_cds.py",
        "checkpoint": ROOT / "generation" / "cds" / "Model.pth",
        "prefix": "mrose_cds",
        "extra_args": [
            "--all_csv",
            "mrose_cds_all_scored.csv",
            "--top_csv",
            "mrose_cds_top.csv",
            "--top_fasta",
            "mrose_cds_top.fasta",
            "--mfe_weight",
            "0",
        ],
    },
    "3utr": {
        "label": "3' UTR",
        "script": ROOT / "generation" / "3utr" / "generate_3utr.py",
        "checkpoint": ROOT / "generation" / "3utr" / "Model.pth",
        "prefix": "mrose_3utr",
        "extra_args": ["--output_prefix", "mrose_3utr"],
    },
}


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        with path.open("rb") as checkpoint:
            header = checkpoint.read(128)
        return header.startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def checkpoint_status() -> dict[str, dict[str, Any]]:
    status = {}
    for region, cfg in REGIONS.items():
        checkpoint = cfg["checkpoint"]
        exists = checkpoint.exists()
        lfs_pointer = is_lfs_pointer(checkpoint)
        status[region] = {
            "exists": exists,
            "ready": exists and not lfs_pointer,
            "lfs_pointer": lfs_pointer,
            "bytes": checkpoint.stat().st_size if exists else 0,
            "path": str(checkpoint),
        }
    return status


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_sequence(sequence: str) -> str:
    cleaned = re.sub(r"\s+", "", sequence).upper().replace("U", "T")
    if not cleaned:
        raise ValueError("Sequence cannot be empty.")
    if len(cleaned) > MAX_SEQUENCE_LENGTH:
        raise ValueError(f"Sequence length exceeds {MAX_SEQUENCE_LENGTH} nt.")
    if not VALID_SEQUENCE.fullmatch(sequence):
        raise ValueError("Sequence may contain only A, C, G, T or U characters.")
    return cleaned


def validate_request(
    region: str,
    sequence: str,
    num_samples: int,
    top_k: int,
    sequence_5utr: str | None = None,
    sequence_cds: str | None = None,
    sequence_3utr: str | None = None,
) -> str | dict[str, str]:
    if region not in REGIONS and region != "full":
        raise ValueError(f"Unsupported region: {region}.")
    if num_samples < 1 or num_samples > MAX_SAMPLES:
        raise ValueError(f"num_samples must be between 1 and {MAX_SAMPLES}.")
    if top_k < 1 or top_k > MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")
    if top_k > num_samples:
        raise ValueError("top_k cannot exceed num_samples.")
    if region == "full":
        parts = {
            "5utr": normalize_sequence(sequence_5utr or sequence),
            "cds": normalize_sequence(sequence_cds or sequence),
            "3utr": normalize_sequence(sequence_3utr or sequence),
        }
        if len(parts["cds"]) < 3:
            raise ValueError("CDS sequence must contain at least one codon.")
        total_length = sum(len(part) for part in parts.values())
        if total_length > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"Full mRNA sequence length exceeds {MAX_SEQUENCE_LENGTH} nt.")
        return parts
    if region == "cds" and len(normalize_sequence(sequence)) < 3:
        raise ValueError("CDS sequence must contain at least one codon.")
    return normalize_sequence(sequence)


def status_path(job_id: str) -> Path:
    return JOB_ROOT / job_id / "status.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def read_status(job_id: str) -> dict[str, Any] | None:
    path = status_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def list_result_files(job_id: str) -> list[str]:
    job_dir = JOB_ROOT / job_id
    out_dir = JOB_ROOT / job_id / "results"
    files = []
    if out_dir.exists():
        files.extend(path.name for path in out_dir.iterdir() if path.is_file())
    files.extend(path.name for path in job_dir.glob("*.log") if path.is_file())
    return sorted(set(files))


def result_file(job_id: str, filename: str) -> Path:
    safe_name = Path(filename).name
    candidates = [
        JOB_ROOT / job_id / "results" / safe_name,
        JOB_ROOT / job_id / safe_name,
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(filename)


def create_job(
    *,
    region: str,
    sequence: str,
    num_samples: int,
    top_k: int,
    device: str,
    temperature: float,
    match_input_length: bool,
    sequence_5utr: str | None = None,
    sequence_cds: str | None = None,
    sequence_3utr: str | None = None,
) -> dict[str, Any]:
    cleaned = validate_request(
        region,
        sequence,
        num_samples,
        top_k,
        sequence_5utr=sequence_5utr,
        sequence_cds=sequence_cds,
        sequence_3utr=sequence_3utr,
    )
    if not re.fullmatch(r"(cpu|cuda(?::\d+)?)", device):
        raise ValueError("device must be cpu, cuda or cuda:<index>.")
    if temperature <= 0 or temperature > 5:
        raise ValueError("temperature must be greater than 0 and no more than 5.")

    job_id = uuid.uuid4().hex
    job_dir = JOB_ROOT / job_id
    output_dir = job_dir / "results"
    job_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    if region == "full":
        assert isinstance(cleaned, dict)
        input_fasta: Path | dict[str, Path] = {}
        for part, part_sequence in cleaned.items():
            path = job_dir / f"input_{part}.fasta"
            path.write_text(f">mrose_input_{part}\n{part_sequence}\n")
            input_fasta[part] = path
        sequence_length: int | dict[str, int] = {
            part: len(part_sequence)
            for part, part_sequence in cleaned.items()
        }
        total_sequence_length = sum(sequence_length.values())
        region_label = "Full-length mRNA"
    else:
        assert isinstance(cleaned, str)
        input_fasta = job_dir / "input.fasta"
        input_fasta.write_text(f">mrose_input\n{cleaned}\n")
        sequence_length = len(cleaned)
        total_sequence_length = len(cleaned)
        region_label = REGIONS[region]["label"]

    status = {
        "job_id": job_id,
        "status": "queued",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "region": region,
        "region_label": region_label,
        "sequence_length": total_sequence_length,
        "sequence_lengths": sequence_length,
        "num_samples": num_samples,
        "top_k": top_k,
        "device": device,
        "temperature": temperature,
        "files": [],
        "error": None,
    }
    write_json(status_path(job_id), status)

    EXECUTOR.submit(
        run_job,
        job_id,
        region,
        input_fasta,
        output_dir,
        num_samples,
        top_k,
        device,
        temperature,
        match_input_length,
    )
    return status


def update_status(job_id: str, **fields: Any) -> None:
    status = read_status(job_id)
    if status is None:
        return
    status.update(fields)
    status["updated_at"] = utc_now()
    write_json(status_path(job_id), status)


def build_command(
    region: str,
    input_fasta: Path,
    output_dir: Path,
    num_samples: int,
    top_k: int,
    device: str,
    temperature: float,
    match_input_length: bool,
) -> list[str]:
    cfg = REGIONS[region]
    command_num_samples = max(num_samples, 2) if region == "5utr" else num_samples
    cmd = [
        PYTHON,
        str(cfg["script"]),
        "--checkpoint",
        str(cfg["checkpoint"]),
        "--input_fasta",
        str(input_fasta),
        "--output_dir",
        str(output_dir),
        "--num_samples",
        str(command_num_samples),
        "--top_k",
        str(top_k),
        "--device",
        device,
        "--temperature",
        str(temperature),
        *cfg["extra_args"],
    ]
    if region == "3utr" and match_input_length:
        cmd.append("--match_input_length")
    if region == "cds":
        cmd.extend(["--batch_size", "32"])
    return cmd


def read_top_sequences(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Expected result file was not created: {path.name}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "sequence" not in rows[0]:
        raise ValueError(f"Result file does not contain a sequence column: {path.name}")
    return rows


def top_csv_path(region: str, output_dir: Path, top_k: int) -> Path:
    if region == "cds":
        return output_dir / "mrose_cds_top.csv"
    return output_dir / f"mrose_{region}_top{top_k}.csv"


def copy_component_outputs(full_output_dir: Path, component_dir: Path, region: str, top_k: int) -> None:
    csv_path = top_csv_path(region, component_dir, top_k)
    fasta_path = component_dir / (
        "mrose_cds_top.fasta" if region == "cds" else f"mrose_{region}_top{top_k}.fasta"
    )
    if csv_path.exists():
        (full_output_dir / f"mrose_full_{region}_top.csv").write_text(csv_path.read_text())
    if fasta_path.exists():
        (full_output_dir / f"mrose_full_{region}_top.fasta").write_text(fasta_path.read_text())


def merge_full_mrna_outputs(output_dir: Path, top_k: int) -> None:
    component_rows = {
        region: read_top_sequences(top_csv_path(region, output_dir / region, top_k))
        for region in ("5utr", "cds", "3utr")
    }
    merge_count = min(top_k, *(len(rows) for rows in component_rows.values()))
    if merge_count < 1:
        raise ValueError("No full-length mRNA candidates could be merged.")

    merged_rows = []
    for index in range(merge_count):
        five = component_rows["5utr"][index]["sequence"].upper().replace("U", "T")
        cds = component_rows["cds"][index]["sequence"].upper().replace("U", "T")
        three = component_rows["3utr"][index]["sequence"].upper().replace("U", "T")
        full_sequence = five + cds + three
        merged_rows.append(
            {
                "rank": index + 1,
                "sequence": full_sequence,
                "full_sequence": full_sequence,
                "five_utr_sequence": five,
                "cds_sequence": cds,
                "three_utr_sequence": three,
                "five_utr_length": len(five),
                "cds_length": len(cds),
                "three_utr_length": len(three),
                "full_length": len(full_sequence),
                "five_utr_score": component_rows["5utr"][index].get("final_score", ""),
                "cds_score": component_rows["cds"][index].get("final_score", ""),
                "three_utr_score": component_rows["3utr"][index].get("final_score", ""),
            }
        )

    csv_path = output_dir / "mrose_full_top.csv"
    fasta_path = output_dir / "mrose_full_top.fasta"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged_rows[0].keys()))
        writer.writeheader()
        writer.writerows(merged_rows)
    with fasta_path.open("w") as handle:
        for row in merged_rows:
            handle.write(
                f">mrose_full_rank_{row['rank']}|"
                f"5utr={row['five_utr_length']}|"
                f"cds={row['cds_length']}|"
                f"3utr={row['three_utr_length']}|"
                f"full={row['full_length']}\n"
            )
            handle.write(str(row["full_sequence"]) + "\n")

    for region in ("5utr", "cds", "3utr"):
        copy_component_outputs(output_dir, output_dir / region, region, top_k)


def run_command(cmd: list[str], stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            check=False,
            text=True,
        )
    return result.returncode


def run_job(
    job_id: str,
    region: str,
    input_fasta: Path,
    output_dir: Path,
    num_samples: int,
    top_k: int,
    device: str,
    temperature: float,
    match_input_length: bool,
) -> None:
    start = time.time()
    try:
        if region == "full":
            assert isinstance(input_fasta, dict)
            commands = {}
            for part in ("5utr", "cds", "3utr"):
                part_output_dir = output_dir / part
                part_output_dir.mkdir(parents=True, exist_ok=True)
                cmd = build_command(
                    part,
                    input_fasta[part],
                    part_output_dir,
                    num_samples,
                    top_k,
                    device,
                    temperature,
                    match_input_length,
                )
                commands[part] = cmd
                update_status(
                    job_id,
                    status="running",
                    stage=f"generating {REGIONS[part]['label']}",
                    commands=commands,
                    started_at=read_status(job_id).get("started_at", utc_now()) if read_status(job_id) else utc_now(),
                )
                returncode = run_command(
                    cmd,
                    JOB_ROOT / job_id / f"stdout_{part}.log",
                    JOB_ROOT / job_id / f"stderr_{part}.log",
                )
                if returncode != 0:
                    raise RuntimeError(f"{REGIONS[part]['label']} generation exited with code {returncode}.")
            update_status(job_id, stage="merging full-length mRNA candidates")
            merge_full_mrna_outputs(output_dir, top_k)
            returncode = 0
        else:
            assert isinstance(input_fasta, Path)
            cmd = build_command(
                region,
                input_fasta,
                output_dir,
                num_samples,
                top_k,
                device,
                temperature,
                match_input_length,
            )
            update_status(job_id, status="running", command=cmd, started_at=utc_now())
            returncode = run_command(cmd, JOB_ROOT / job_id / "stdout.log", JOB_ROOT / job_id / "stderr.log")
        fields: dict[str, Any] = {
            "returncode": returncode,
            "duration_seconds": round(time.time() - start, 2),
            "files": list_result_files(job_id),
            "finished_at": utc_now(),
        }
        if returncode == 0:
            fields["status"] = "succeeded"
            fields["stage"] = "complete"
        else:
            fields["status"] = "failed"
            fields["error"] = f"Generation command exited with code {returncode}."
        update_status(job_id, **fields)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        update_status(
            job_id,
            status="failed",
            error=str(exc),
            duration_seconds=round(time.time() - start, 2),
            finished_at=utc_now(),
            files=list_result_files(job_id),
        )
