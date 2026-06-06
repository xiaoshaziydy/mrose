# CDS vaccine mRNA degradation regression example

**Task type:** regression.  
**Biological target:** degradation or stability-associated quantitative value for CDS-derived mRNA sequences.  
**Source-style data:** CSV with `Sequence`, `Value`, `Dataset` and `Split` columns. The training script consumes `Sequence` and `Value`.

```text
experiments/cds_degradation/example/
├── Data/
│   ├── train.csv
│   └── test.csv
└── run_example.sh
```

Run from the repository root:

```bash
bash experiments/cds_degradation/example/run_example.sh
```

The current CDS script uses DDP-style launch and expects the data folder to be named `Data` relative to the current working directory. The example wrapper preserves that convention.
