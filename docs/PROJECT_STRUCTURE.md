# Project structure

The repository separates importable model code from task-specific experiment launchers.

## `mrose/`

Core modules:

- `module_5utr.py`: 5′UTR sequence model and 5′UTR feature extraction.
- `module_cds.py`: CDS model with codon-level and nucleotide-level processing.
- `module_3utr.py`: 3′UTR sequence model and 3′UTR feature extraction.
- `full_length_fusion.py`: full-length mRNA fusion model that combines 5′UTR, CDS and 3′UTR branches.

## `experiments/`

Each task folder contains:

- the original or task-specific training script;
- `example/` data showing the expected input schema;
- a local README and runnable wrapper where possible.

Task folders are named by biological region and task type rather than only by model module.

## `generation/`

Standalone candidate-generation scripts:

- `5utr/generate_5utr.py`: 5′ UTR generation and ranking.
- `cds/generate_cds.py`: CDS generation, amino-acid preservation checks and ranking.
- `3utr/generate_3utr.py`: 3′ UTR generation and ranking.
- `examples/`: compact FASTA templates used by `scripts/demo_generate_sequences.py`.

Released generation checkpoints are expected as `generation/5utr/Model.pth`, `generation/cds/Model.pth` and `generation/3utr/Model.pth`, and are tracked with Git LFS.
