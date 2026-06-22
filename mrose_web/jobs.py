"""Background job helpers for the mROSE FastAPI service."""

from __future__ import annotations

import json
import os
import re
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


def validate_request(region: str, sequence: str, num_samples: int, top_k: int) -> str:
    if region not in REGIONS:
        raise ValueError(f"Unsupported region: {region}.")
    if num_samples < 1 or num_samples > MAX_SAMPLES:
        raise ValueError(f"num_samples must be between 1 and {MAX_SAMPLES}.")
    if top_k < 1 or top_k > MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")
    if top_k > num_samples:
        raise ValueError("top_k cannot exceed num_samples.")
    if region == "cds" and len(normalize_sequence(sequence)) < 3:
        raise ValueError("CDS sequence must contain at least one codon.")
    return normalize_sequence(sequence)


def status_path(job_id: str) -> Path:
    return JOB_ROOT / job_id / "status.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
    tmp.replace(path)


def read_status(job_id: str) -> dict[str, Any] | None:
    path = status_path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_result_files(job_id: str) -> list[str]:
    job_dir = JOB_ROOT / job_id
    out_dir = JOB_ROOT / job_id / "results"
    files = []
    if out_dir.exists():
        files.extend(path.name for path in out_dir.iterdir() if path.is_file())
    for log_name in ("stdout.log", "stderr.log"):
        if (job_dir / log_name).exists():
            files.append(log_name)
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
) -> dict[str, Any]:
    cleaned = validate_request(region, sequence, num_samples, top_k)
    if not re.fullmatch(r"(cpu|cuda(?::\d+)?)", device):
        raise ValueError("device must be cpu, cuda or cuda:<index>.")
    if temperature <= 0 or temperature > 5:
        raise ValueError("temperature must be greater than 0 and no more than 5.")

    job_id = uuid.uuid4().hex
    job_dir = JOB_ROOT / job_id
    input_fasta = job_dir / "input.fasta"
    output_dir = job_dir / "results"
    job_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_fasta.write_text(f">mrose_input\\n{cleaned}\\n")

    status = {
        "job_id": job_id,
        "status": "queued",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "region": region,
        "region_label": REGIONS[region]["label"],
        "sequence_length": len(cleaned),
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
        str(num_samples),
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
    stdout_path = JOB_ROOT / job_id / "stdout.log"
    stderr_path = JOB_ROOT / job_id / "stderr.log"
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
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                check=False,
                text=True,
            )
        fields: dict[str, Any] = {
            "returncode": result.returncode,
            "duration_seconds": round(time.time() - start, 2),
            "files": list_result_files(job_id),
            "finished_at": utc_now(),
        }
        if result.returncode == 0:
            fields["status"] = "succeeded"
        else:
            fields["status"] = "failed"
            fields["error"] = f"Generation command exited with code {result.returncode}."
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
