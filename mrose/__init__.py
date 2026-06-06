"""mROSE: region-aware mRNA optimization models.

Core modules:
- Model_5UTR for 5′ UTR representation learning
- Model_CDS for codon/nucleotide dual-stream CDS modelling
- Model_3UTR for 3′ UTR regulatory modelling
- GlobalmRNAMasterModel for full-length fusion
"""

from .module_5utr import Model_5UTR, compute_features_5utr
from .module_cds import Model_CDS, DEFAULT_CODON_TABLE
from .module_3utr import Model_3UTR, compute_features_3utr
from .full_length_fusion import GlobalmRNAMasterModel, FullmRNAFusionDataset

__all__ = [
    "Model_5UTR",
    "compute_features_5utr",
    "Model_CDS",
    "DEFAULT_CODON_TABLE",
    "Model_3UTR",
    "compute_features_3utr",
    "GlobalmRNAMasterModel",
    "FullmRNAFusionDataset",
]
