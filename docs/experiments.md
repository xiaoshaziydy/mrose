# Experiments

The compact example datasets document expected schemas, loader conventions and command-line usage.

| Task family | Folder | Input format | Label type |
|---|---|---|---|
| 5′ UTR MRL regression | `experiments/5utr_mrl/` | CSV with `utr` and `rl` | continuous |
| CDS degradation regression | `experiments/cds_degradation/` | CSV with `Sequence` and `Value` | continuous |
| 3′ UTR RBP classification | `experiments/3utr_rbp/` | FASTA headers containing `class_0` or `class_1` | binary |
| Full-length mRNA stability regression | `experiments/full_length_stability/` | CSV with `sequence` and `label` | continuous |

Run the compact examples from the repository root:

```bash
bash experiments/5utr_mrl/example/run_example.sh
bash experiments/cds_degradation/example/run_example.sh
bash experiments/3utr_rbp/example/AKAP1_HepG2/run_example.sh
bash experiments/full_length_stability/example/run_example.sh
```

Full-scale training data and non-release model artifacts should be stored outside normal Git history. See `DATA_MANIFEST.md` and `data/README.md` for dataset placement notes.
