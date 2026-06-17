# Data manifest

Large source datasets are excluded from this GitHub-ready repository. Miniature examples are included under each experiment folder to document the required schema.

| Task | Example location | Source-style file(s) | Required columns or labels | Notes |
|---|---|---|---|---|
| 5′UTR MRL regression | `experiments/5utr_mrl/example/` | `5UTR/MRL/*_BiologyFeatures.csv` | `utr`, `rl` | The source file includes additional ribosome-loading and biological feature columns. |
| CDS degradation regression | `experiments/cds_degradation/example/Data/` | `CDS/CoV_Vaccine_Degradation.csv` | `Sequence`, `Value`, `Dataset`, `Split` | The training script consumes `Sequence` and `Value`. |
| 3′UTR RBP classification | `experiments/3utr_rbp/example/AKAP1_HepG2/` | `3UTR/RNA_protein_interaction/<RBP_cell>/fold*/{train,dev}.fa`, `test.fa` | FASTA header contains `class_0` or `class_1` | Example uses a small AKAP1_HepG2 subset. |
| Full-length mRNA stability regression | `experiments/full_length_stability/example/in_solution_half_life/` | `Full_lenth/in_solution_half_life/{train,dev,test}.csv` | `sequence`, `label` | The loader segments full transcripts into 5′UTR, CDS and 3′UTR. |
| Sequence generation | `generation/examples/` | `Generate.rar` local bundle | FASTA template sequences plus region-specific `.pth` checkpoints | Released checkpoints are tracked with Git LFS as `generation/5utr/Model.pth`, `generation/cds/Model.pth` and `generation/3utr/Model.pth`. |

Recommended full-data layout for local training:

```text
data/raw/
├── 5utr_mrl/
├── cds_degradation/
├── 3utr_rbp/
└── full_length_stability/
```

Released checkpoint layout for generation:

```text
generation/
├── 5utr/Model.pth
├── cds/Model.pth
└── 3utr/Model.pth
```

`data/raw/`, non-release model checkpoints and generated outputs are ignored by `.gitignore`. Released generation checkpoints are tracked with Git LFS.
