# Sequence generation

This folder contains standalone mROSE sequence-generation entry points for the
three major mRNA regions and full-length assembly:

- `5utr/generate_5utr.py`: generate and rank 5′ UTR candidates.
- `cds/generate_cds.py`: generate length-matched CDS candidates and rank them
  with model score, CAI, GC and optional MFE.
- `3utr/generate_3utr.py`: generate and rank 3′ UTR candidates.
- `full_length/generate_full_length.py`: two input modes for full-length mRNA
  generation (see below).

The scripts were packaged from the local generation bundle and are designed to
use one checkpoint per region for both candidate generation and scoring.

For an interactive walkthrough with pre-computed results, open
[notebooks/mROSE_generation.ipynb](../notebooks/mROSE_generation.ipynb).

## Checkpoints

The released generation checkpoints are tracked with Git LFS. After cloning the
repository, run:

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

If Git LFS is not installed, these paths may contain small pointer files instead
of the real checkpoints, and generation will fail when PyTorch tries to load
them.

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

### 5′ UTR

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

### CDS

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

### 3′ UTR

```bash
python generation/3utr/generate_3utr.py \
  --checkpoint generation/3utr/Model.pth \
  --input_fasta generation/examples/3utr_template.fasta \
  --num_samples 20 \
  --top_k 5 \
  --device cpu \
  --output_dir outputs/generation/3utr_example \
  --output_prefix example_3utr
```

By default, 3′ UTR candidates are allowed to vary in length. Add
`--match_input_length` only if you specifically need legacy input-length
matching.

### Full-length mRNA

The full-length launcher supports **two input modes**:

#### Mode 1: Single full-length mRNA (recommended)

Provide a single FASTA containing a full-length mRNA sequence. The script
**automatically splits** it into 5′ UTR, CDS and 3′ UTR by ORF detection
(ATG → first in-frame stop codon), then runs each regional generator:

```bash
python generation/full_length/generate_full_length.py \
  --full_mrna_fasta generation/examples/full_mrna_template.fasta \
  --num_samples 20 \
  --top_k 5 \
  --device cpu \
  --output_dir outputs/generation/full_length_example \
  --output_prefix example_full_length
```

#### Mode 2: Three separate regional FASTAs (legacy)

Provide separate FASTA files for each region:

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

**How splitting works** (Mode 1):

```
Input: AGGAATAA...CCACC ATG ... CDS ... TAA GCTGCC...TAAA
                     |      |            |
                     5′ UTR  CDS         3′ UTR
```

- 5′ UTR = everything before the first ATG
- CDS = from ATG through the first downstream in-frame stop codon
- 3′ UTR = everything after the stop codon

Fallbacks: if no ATG is found, the first third is used as 5′ UTR. If no in-frame
stop codon is found, the CDS is truncated to a length divisible by 3.

**How merging works** (both modes):

```
full_length_rank_1 = 5utr_generated_rank_1 + cds_generated_rank_1 + 3utr_generated_rank_1
full_length_rank_2 = 5utr_generated_rank_2 + cds_generated_rank_2 + 3utr_generated_rank_2
  ...
```

The full-length launcher writes regional outputs under:

```text
outputs/generation/full_length_example/
├── split_input/                       # (Mode 1 only) split regional FASTAs
│   ├── example_full_length_5utr_split.fasta
│   ├── example_full_length_cds_split.fasta
│   └── example_full_length_3utr_split.fasta
├── 5utr/
├── cds/
├── 3utr/
├── example_full_length_top5.csv
└── example_full_length_top5.fasta
```

**Output columns** (CSV): `rank`, `sequence`, `five_utr_sequence`,
`cds_sequence`, `three_utr_sequence`, `five_utr_length`, `cds_length`,
`three_utr_length`, `full_length`, `five_utr_score`, `cds_score`,
`three_utr_score`.

## Runtime notes

The generation scripts require the scientific Python stack used by mROSE,
including PyTorch, NumPy, pandas, SciPy, scikit-learn, tqdm and Biopython. The
5′ UTR and 3′ UTR generators also require ViennaRNA Python bindings for MFE
scoring. The CDS example disables MFE scoring by default with `--mfe_weight 0`
so it can run in environments without ViennaRNA.
