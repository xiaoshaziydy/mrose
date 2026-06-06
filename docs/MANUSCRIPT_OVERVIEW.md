# Manuscript overview

This repository is organized around the manuscript **"mROSE: mRNA Optimization for Stability and Expression with a diffusion-based generative framework"**.

## Core idea

mROSE treats mRNA optimization as a region-aware sequence generation problem. Instead of relying only on predefined rules or ranking a fixed candidate pool, the framework learns latent sequence-function neighborhoods and uses conditional denoising to explore optimized variants.

The model separates each transcript into three regulatory regions:

- **5′ UTR**, modeled at nucleotide resolution to capture translation-initiation motifs, positional signals and local sequence context.
- **CDS**, modeled with coupled codon-level and nucleotide-level representations so synonymous sequence space can be searched while preserving protein identity.
- **3′ UTR**, modeled at nucleotide resolution to capture post-transcriptional regulatory motifs, RNA-binding protein signals and modification-associated sequence patterns.

The same region-aware design principle supports both modular optimization and full-length mRNA assembly.

## Architecture summary

mROSE contains four conceptual stages:

1. Region-specific encoders transform 5′ UTR, CDS and 3′ UTR sequences into latent representations.
2. A forward diffusion process perturbs the latent states to encourage robust functional representations.
3. A task-conditioned Transformer denoising U-Net reconstructs informative latent states under molecular objectives such as expression, degradation, mean ribosome loading, RBP interaction and m6A prediction.
4. Region-specific design heads decode optimized regional candidates that can be evaluated alone or assembled into full-length transcripts.

## Benchmarks described in the manuscript

The manuscript evaluates mROSE across several task families:

- CDS-associated expression and degradation prediction.
- SARS-CoV-2 vaccine mRNA degradation prediction.
- Riboswitch activity prediction.
- 5′ UTR translational-control benchmarks.
- 3′ UTR RBP interaction and m6A-related prediction.
- Full-length mRNA stability and expression-associated phenotypes.
- Broader regulatory sequence tasks such as promoter, splice-site and chromatin-associated prediction.

Across these settings, mROSE is reported to outperform specialized predictors and genomic foundation-model baselines in the manuscript.

## Experimental validation described in the manuscript

The experimental workflow validates mROSE-designed 5′ UTRs, CDSs and 3′ UTRs in reporter systems, then assembles optimized modules into full-length mRNAs. In the reported in vivo study, lipid nanoparticle-formulated mROSE-designed erythropoietin mRNA increased serum protein expression in mice by up to 15.3-fold compared with the benchmark transcript.

## Repository mapping

| Manuscript component | Repository location |
|---|---|
| 5′ UTR modeling | `mrose/module_5utr.py`, `experiments/5utr_mrl/` |
| CDS modeling | `mrose/module_cds.py`, `experiments/cds_degradation/` |
| 3′ UTR modeling | `mrose/module_3utr.py`, `experiments/3utr_rbp/` |
| Full-length mRNA fusion | `mrose/full_length_fusion.py`, `experiments/full_length_stability/` |
| Data placement notes | `DATA_MANIFEST.md`, `data/README.md` |
| Quick import check | `scripts/quick_import_check.py` |

## Important note

This repository contains code organization and compact examples. Full benchmark datasets, trained checkpoints and generated outputs are intentionally excluded from normal Git history and should be distributed through an appropriate data or model repository when licensing permits.
