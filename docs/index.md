# mROSE documentation

```{image} assets/mrose-icon.png
:alt: mROSE icon
:width: 180px
:align: center
```

**mROSE** is a region-aware diffusion framework for optimizing messenger RNA (mRNA) sequences for improved stability and protein expression. It accompanies the manuscript **"mROSE: mRNA Optimization for Stability and Expression with a diffusion-based generative framework"**.

```{image} assets/mrose-figure1.png
:alt: Overview of the mROSE region-aware mRNA optimization workflow
:width: 900px
:align: center
```

mROSE separates mRNA design into biologically meaningful regions:

- **5′ UTR module** for translation initiation and mean ribosome loading prediction.
- **CDS module** for coding-region expression and degradation tasks using coupled codon-level and nucleotide-level representations.
- **3′ UTR module** for post-transcriptional regulation, including RNA-binding protein interaction and m6A-related regulatory prediction.
- **Full-length fusion module** for transcript-level stability and expression-associated phenotypes.

This documentation explains how to install the project, run compact examples, use the sequence-generation demos and map the codebase to the manuscript.

```{toctree}
:maxdepth: 2
:caption: User guide

installation
generation
notebooks/mROSE_generation_demo
experiments
PROJECT_STRUCTURE
MANUSCRIPT_OVERVIEW
```

```{toctree}
:maxdepth: 1
:caption: Project links

GitHub repository <https://github.com/xiaoshaziydy/mrose>
```
