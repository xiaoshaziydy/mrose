# Sequence generation

This folder contains standalone mROSE sequence-generation entry points for the three major mRNA regions:

- `5utr/generate_5utr.py`: generate and rank 5′ UTR candidates.
- `cds/generate_cds.py`: generate length-matched CDS candidates and rank them with model score, CAI, GC and optional MFE.
- `3utr/generate_3utr.py`: generate and rank 3′ UTR candidates.

The scripts were packaged from the local generation bundle and are designed to use one checkpoint per region for both candidate generation and scoring.

## Checkpoints

The released generation checkpoints are tracked with Git LFS. After cloning the repository, run:

```bash
git lfs install
git lfs pull
```

The files should be present here:

```text
generation/
├── 5utr/Model.pth
├── cds/Model.pth
└── 3utr/Model.pth
```

Verify checkpoint integrity:

```bash
shasum -a 256 -c MODEL_CHECKSUMS.sha256
```

If Git LFS is not installed, these paths may contain small pointer files instead of the real checkpoints, and generation will fail when PyTorch tries to load them.

## Demo

Print the dependency and checkpoint status plus ready-to-run commands:

```bash
python scripts/demo_generate_sequences.py
```

Run one demo task:

```bash
python scripts/demo_generate_sequences.py --run 5utr
python scripts/demo_generate_sequences.py --run cds
python scripts/demo_generate_sequences.py --run 3utr
```

Run all demo tasks:

```bash
python scripts/demo_generate_sequences.py --run all
```

Outputs are written under `outputs/generation/`, which is ignored by Git.

## Direct commands

5′ UTR:

```bash
python generation/5utr/generate_5utr.py \
  --checkpoint generation/5utr/Model.pth \
  --input_fasta generation/examples/5utr_template.fasta \
  --num_samples 20 \
  --top_k 5 \
  --device cpu \
  --output_dir outputs/generation/5utr_demo \
  --output_prefix demo_5utr
```

CDS:

```bash
python generation/cds/generate_cds.py \
  --checkpoint generation/cds/Model.pth \
  --input_fasta generation/examples/cds_template.fasta \
  --num_samples 20 \
  --top_k 5 \
  --device cpu \
  --mfe_weight 0 \
  --output_dir outputs/generation/cds_demo
```

3′ UTR:

```bash
python generation/3utr/generate_3utr.py \
  --checkpoint generation/3utr/Model.pth \
  --input_fasta generation/examples/3utr_template.fasta \
  --num_samples 20 \
  --top_k 5 \
  --device cpu \
  --match_input_length \
  --output_dir outputs/generation/3utr_demo \
  --output_prefix demo_3utr
```

## Runtime notes

The generation scripts require the scientific Python stack used by mROSE, including PyTorch, NumPy, pandas, SciPy, scikit-learn, tqdm and Biopython. The 5′ UTR and 3′ UTR generators also require ViennaRNA Python bindings for MFE scoring. The CDS demo disables MFE scoring by default with `--mfe_weight 0` so it can run in environments without ViennaRNA.
