# Data organization

Large training datasets are not committed to this GitHub-ready package. Task-specific miniature examples are stored inside each experiment folder so the expected input schema is visible next to the corresponding script:

```text
experiments/5utr_mrl/example/
experiments/cds_degradation/example/
experiments/3utr_rbp/example/
experiments/full_length_stability/example/
```

For full-scale training, place large datasets outside Git or manage them with Git LFS, DVC, Zenodo, Figshare or an institutional data repository. See `DATA_MANIFEST.md` for source-style dataset descriptions.
