from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mrose import (
    Model_5UTR, Model_CDS, Model_3UTR, GlobalmRNAMasterModel,
    compute_features_5utr, compute_features_3utr,
)

print("mROSE imports OK")
print("5UTR feature length:", len(compute_features_5utr("ATGCGTACGTAG")))
print("3UTR feature length:", len(compute_features_3utr("ATGCGTACGTAG")))
print("Core classes:", Model_5UTR.__name__, Model_CDS.__name__, Model_3UTR.__name__, GlobalmRNAMasterModel.__name__)
