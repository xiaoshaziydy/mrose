# 3′ UTR RNA–protein interaction classification example

**Task type:** binary classification.  
**Biological target:** RNA-binding protein interaction sites in 3′UTR-like sequence windows.  
**Source-style data:** FASTA records whose headers contain `class_0` or `class_1`.

```text
experiments/3utr_rbp/example/AKAP1_HepG2/
├── fold0/
├── fold1/
├── fold2/
├── fold3/
├── fold4/
├── test.fa
└── run_example.sh
```

Run from the repository root:

```bash
bash experiments/3utr_rbp/example/AKAP1_HepG2/run_example.sh
```

The current 3′UTR script is DDP-oriented and internally loops over `fold1` to `fold4`, so the example contains all five folds for compatibility with both zero-indexed source data and the script convention.
