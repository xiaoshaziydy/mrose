# Full-length mRNA stability regression example

**Task type:** regression.  
**Biological target:** full-length mRNA phenotype such as in-solution half-life.  
**Source-style data:** each task folder contains `train.csv`, `dev.csv` and `test.csv` with `sequence` and `label` columns.

```text
experiments/full_length_stability/example/in_solution_half_life/
├── train.csv
├── dev.csv
└── test.csv
```

Run from the repository root:

```bash
bash experiments/full_length_stability/example/run_example.sh
```

The full-length loader automatically segments a transcript into 5′UTR, CDS and 3′UTR using the first `ATG` and the first in-frame stop codon after it.
