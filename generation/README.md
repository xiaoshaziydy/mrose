# Sequence generation

This folder contains standalone mROSE sequence-generation entry points for the three major mRNA regions:

- `5utr/generate_5utr.py`: generate and rank 5′ UTR candidates.
- `cds/generate_cds.py`: generate length-matched CDS candidates and rank them with model score, CAI, GC and optional MFE.
- `3utr/generate_3utr.py`: generate and rank 3′ UTR candidates.
- `full_length/generate_full_length.py`: run the three regional generators and merge same-rank 5′ UTR, CDS and 3′ UTR candidates into full-length mRNA candidates.

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

## Example

Print the dependency and checkpoint status plus ready-to-run commands:

```bash
python scripts/generate_sequences.py
```

Run one example task:

```bash
python scripts/generate_sequences.py --run 5utr
python scripts/generate_sequences.py --run cds
python scripts/generate_sequences.py --run 3utr
python scripts/generate_sequences.py --run full_length
```

Run all example tasks:

```bash
python scripts/generate_sequences.py --run all
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
  --output_dir outputs/generation/5utr_example \
  --output_prefix example_5utr
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
  --output_dir outputs/generation/cds_example
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
  --output_dir outputs/generation/3utr_example \
  --output_prefix example_3utr
```

Full-length mRNA:

```bash
python generation/full_length/generate_full_length.py \
  --five_utr_fasta generation/examples/5utr_template.fasta \
  --cds_fasta generation/examples/cds_template.fasta \
  --three_utr_fasta generation/examples/3utr_template.fasta \
  --num_samples 20 \
  --top_k 5 \
  --device cpu \
  --output_dir outputs/generation/full_length_example \
  --output_prefix example_full_length
```

The full-length launcher writes regional outputs under:

```text
outputs/generation/full_length_example/
├── 5utr/
├── cds/
├── 3utr/
├── example_full_length_top5.csv
└── example_full_length_top5.fasta
```

Rows are merged by rank: rank 1 5′ UTR + rank 1 CDS + rank 1 3′ UTR, rank 2 + rank 2 + rank 2, and so on.

## Runtime notes

The generation scripts require the scientific Python stack used by mROSE, including PyTorch, NumPy, pandas, SciPy, scikit-learn, tqdm and Biopython. The 5′ UTR and 3′ UTR generators also require ViennaRNA Python bindings for MFE scoring. The CDS example disables MFE scoring by default with `--mfe_weight 0` so it can run in environments without ViennaRNA.
