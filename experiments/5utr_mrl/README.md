# 5′ UTR MRL regression example

**Task type:** regression.  
**Biological target:** mean ribosome loading or related translation output from 5′ UTR sequence.  
**Source-style data:** CSV file with at least `utr` and `rl` columns. The included example retains the original feature columns from the source 5′UTR MRL table.

```text
experiments/5utr_mrl/example/
├── train.csv
├── test.csv
└── run_example.sh
```

Run from the repository root:

```bash
bash experiments/5utr_mrl/example/run_example.sh
```

For real training, replace `example/train.csv` and `example/test.csv` with full MRL library files and use a CUDA device, for example `--device cuda:0`.
