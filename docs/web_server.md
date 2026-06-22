# Web server

mROSE includes a lightweight FastAPI web server for online region-specific sequence generation.

The server provides:

- a browser form at `/`;
- `GET /api/health` for service and checkpoint status;
- `POST /api/generate` to submit 5′ UTR, CDS or 3′ UTR generation jobs;
- `GET /api/jobs/{job_id}` to poll job status;
- `GET /api/jobs/{job_id}/files/{filename}` to download CSV, FASTA and log files.

Jobs run in the background and write isolated outputs under `outputs/web_jobs/`. This directory is ignored by Git.

## Install

Install the mROSE scientific environment first, then add the web dependencies:

```bash
pip install -r web-requirements.txt
```

If PyTorch CUDA wheels are needed, install the main requirements as described in the project README before installing the web layer.

## Run locally

From the repository root:

```bash
uvicorn mrose_web.app:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

The OpenAPI schema is available at:

```text
http://localhost:8000/docs
```

## Runtime configuration

The server reads these environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `MROSE_WEB_JOB_DIR` | `outputs/web_jobs` | Job input, output, status and log directory |
| `MROSE_WEB_PYTHON` | current Python | Python interpreter used to launch generation scripts |
| `MROSE_WEB_MAX_WORKERS` | `1` | Number of background generation jobs to run at once |
| `MROSE_WEB_MAX_SEQUENCE_LENGTH` | `6000` | Maximum accepted input sequence length |
| `MROSE_WEB_MAX_SAMPLES` | `10000` | Maximum accepted `num_samples` |
| `MROSE_WEB_MAX_TOP_K` | `100` | Maximum accepted `top_k` |

For the project server environment, a typical command is:

```bash
MROSE_WEB_PYTHON=/root/miniconda3/envs/DiffRNA/bin/python \
MROSE_WEB_MAX_WORKERS=1 \
uvicorn mrose_web.app:app --host 0.0.0.0 --port 8000
```

## Production deployment

For a single-machine deployment, run the API behind Nginx:

```bash
gunicorn mrose_web.app:app \
  -k uvicorn.workers.UvicornWorker \
  --bind 127.0.0.1:8000 \
  --workers 1 \
  --timeout 3600
```

Use one worker when jobs use a single GPU. Increase `MROSE_WEB_MAX_WORKERS` only if the server has enough GPU or CPU capacity.

Example Nginx reverse proxy:

```nginx
server {
    listen 80;
    server_name example.org;

    client_max_body_size 2m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 3600;
    }
}
```

Use HTTPS in production, for example with Certbot-managed TLS certificates.

## API example

Submit a job:

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "region": "5utr",
    "sequence": "AGGAATAAACTAGTATTCTTCTGGTCCCCACAGACTCAGAGAGAACCCGCCACC",
    "num_samples": 100,
    "top_k": 10,
    "device": "cuda:0",
    "temperature": 1.0
  }'
```

Poll the returned job:

```bash
curl http://localhost:8000/api/jobs/<job_id>
```

Download a result file:

```bash
curl -O http://localhost:8000/api/jobs/<job_id>/files/mrose_5utr_top10.csv
```

## Safety notes

The web layer validates region, device, sequence characters, sequence length, sample count and top-k before launching a job. Generation commands are executed as argument lists, not through a shell. Keep the server behind authentication or a private network if it is attached to a GPU host, because each request can consume substantial compute.

