# Data manifest

Large source datasets are excluded from this GitHub-ready repository. Miniature examples are included under each experiment folder to document the required schema.

| Task | Example location | Source-style file(s) | Required columns or labels | Notes |
|---|---|---|---|---|
| 5′UTR MRL regression | `experiments/5utr_mrl/example/` | `5UTR/MRL/*_BiologyFeatures.csv` | `utr`, `rl` | The source file includes additional ribosome-loading and biological feature columns. |
| CDS degradation regression | `experiments/cds_degradation/example/Data/` | `CDS/CoV_Vaccine_Degradation.csv` | `Sequence`, `Value`, `Dataset`, `Split` | The training script consumes `Sequence` and `Value`. |
| 3′UTR RBP classification | `experiments/3utr_rbp/example/AKAP1_HepG2/` | `3UTR/RNA_protein_interaction/<RBP_cell>/fold*/{train,dev}.fa`, `test.fa` | FASTA header contains `class_0` or `class_1` | Example uses a small AKAP1_HepG2 subset. |
| Full-length mRNA stability regression | `experiments/full_length_stability/example/in_solution_half_life/` | `Full_lenth/in_solution_half_life/{train,dev,test}.csv` | `sequence`, `label` | The loader segments full transcripts into 5′UTR, CDS and 3′UTR. |

Recommended full-data layout for local training:

```text
data/raw/
├── 5utr_mrl/
├── cds_degradation/
├── 3utr_rbp/
└── full_length_stability/
```

`data/raw/`, model checkpoints and generated outputs are ignored by `.gitignore`.
