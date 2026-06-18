# Sequence generation

mROSE includes standalone generation entry points for designing region-specific candidate sequences.

| Region | Script | Output |
|---|---|---|
| 5′ UTR | `generation/5utr/generate_5utr.py` | scored 5′ UTR candidates with predicted MRL, MFE, GC and uORF/uAUG features |
| CDS | `generation/cds/generate_cds.py` | length-matched CDS candidates ranked by model score, CAI, GC and optional MFE |
| 3′ UTR | `generation/3utr/generate_3utr.py` | scored 3′ UTR candidates with degradation-style prediction, MFE, GC/TC and motif penalties |

Print dependency and checkpoint status plus ready-to-run commands:

```bash
python scripts/demo_generate_sequences.py
```

Run one compact demo:

```bash
python scripts/demo_generate_sequences.py --run 5utr
python scripts/demo_generate_sequences.py --run cds
python scripts/demo_generate_sequences.py --run 3utr
```

Run all demo tasks:

```bash
python scripts/demo_generate_sequences.py --run all
```

Outputs are written under `outputs/generation/`, which is ignored by Git.

## Direct commands

5′ UTR:

```bash
python generation/5utr/generate_5utr.py \
  --checkpoint generation/5utr/Model.pth \
  --input_fasta generation/examples/5utr_template.fasta \
  --num_samples 20 \
  --top_k 5 \
  --device cpu \
  --output_dir outputs/generation/5utr_demo \
  --output_prefix demo_5utr
```

CDS:

```bash
python generation/cds/generate_cds.py \
  --checkpoint generation/cds/Model.pth \
  --input_fasta generation/examples/cds_template.fasta \
  --num_samples 20 \
  --top_k 5 \
  --device cpu \
  --mfe_weight 0 \
  --output_dir outputs/generation/cds_demo
```

3′ UTR:

```bash
python generation/3utr/generate_3utr.py \
  --checkpoint generation/3utr/Model.pth \
  --input_fasta generation/examples/3utr_template.fasta \
  --num_samples 20 \
  --top_k 5 \
  --device cpu \
  --match_input_length \
  --output_dir outputs/generation/3utr_demo \
  --output_prefix demo_3utr
```

The generation scripts require the scientific Python stack used by mROSE, including PyTorch, NumPy, pandas, SciPy, scikit-learn, tqdm and Biopython. The 5′ UTR and 3′ UTR generators also require ViennaRNA Python bindings for MFE scoring. The CDS demo disables MFE scoring by default with `--mfe_weight 0`.
