"""FastAPI application for online mROSE sequence generation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .jobs import (
    DEFAULT_DEVICE,
    MAX_SAMPLES,
    MAX_SEQUENCE_LENGTH,
    MAX_TOP_K,
    checkpoint_status,
    create_job,
    list_result_files,
    read_status,
    result_file,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"
PRIVATE_STATUS_FIELDS = {"command", "commands", "device"}

app = FastAPI(
    title="mROSE Web Server",
    description="Online API for region-aware mRNA sequence generation with mROSE.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class GenerationRequest(BaseModel):
    region: Literal["5utr", "cds", "3utr", "full"] = Field(..., description="mRNA region to optimize.")
    sequence: str = Field(..., min_length=1, max_length=MAX_SEQUENCE_LENGTH)
    sequence_5utr: str | None = Field(default=None, max_length=MAX_SEQUENCE_LENGTH)
    sequence_cds: str | None = Field(default=None, max_length=MAX_SEQUENCE_LENGTH)
    sequence_3utr: str | None = Field(default=None, max_length=MAX_SEQUENCE_LENGTH)
    num_samples: int = Field(default=100, ge=1, le=MAX_SAMPLES)
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)
    device: str = Field(default=DEFAULT_DEVICE, description="cpu, cuda or cuda:<index>.")
    temperature: float = Field(default=1.0, gt=0, le=5)
    match_input_length: bool = Field(default=False, description="Optional legacy 3' UTR length matching.")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/health")
def health() -> dict[str, object]:
    checkpoints = checkpoint_status()
    ready_for_generation = all(item["ready"] for item in checkpoints.values())
    public_checkpoints = {
        region: {"ready": bool(item["ready"])}
        for region, item in checkpoints.items()
    }
    return {
        "status": "ok",
        "service": "mROSE",
        "checkpoints": public_checkpoints,
        "ready_for_generation": ready_for_generation,
        "limits": {
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "max_samples": MAX_SAMPLES,
            "max_top_k": MAX_TOP_K,
        },
        "regions": {
            "5utr": {"label": "5' UTR", "ready": checkpoints["5utr"]["ready"]},
            "cds": {"label": "CDS", "ready": checkpoints["cds"]["ready"]},
            "3utr": {"label": "3' UTR", "ready": checkpoints["3utr"]["ready"]},
            "full": {"label": "Full-length mRNA", "ready": ready_for_generation},
        },
    }


def public_status(status: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in status.items()
        if key not in PRIVATE_STATUS_FIELDS
    }


@app.post("/api/generate", status_code=202)
def submit_generation(payload: GenerationRequest) -> dict[str, object]:
    try:
        return public_status(create_job(**payload.dict()))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    status = read_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    status["files"] = list_result_files(job_id)
    return public_status(status)


@app.get("/api/jobs/{job_id}/files/{filename}")
def download_result(job_id: str, filename: str) -> FileResponse:
    try:
        path = result_file(job_id, filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="File not found.") from exc
    return FileResponse(path, filename=path.name)
