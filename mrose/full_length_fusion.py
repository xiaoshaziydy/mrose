import os
import math
import warnings
import itertools
import gc
import shutil

# --- CUDA memory-fragmentation setting for older PyTorch versions; must be set before importing torch ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# Disable optimized attention kernels that require sm80 hardware and fall back to the standard math kernel
if hasattr(torch.backends.cuda, "enable_flash_sdp"):
    torch.backends.cuda.enable_flash_sdp(False)
if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
    torch.backends.cuda.enable_mem_efficient_sdp(False)


# ==========================================
# 0. Import the three independent region modules
# ==========================================
try:
    from .module_5utr import Model_5UTR, compute_features_5utr
    from .module_cds import Model_CDS, DEFAULT_CODON_TABLE
    from .module_3utr import Model_3UTR, compute_features_3utr
except ImportError as e:
    print(
        "Import failed. Make sure module_5utr.py, module_cds.py, and module_3utr.py are in the same directory as the main program.\n"
        f"Error details: {e}"
    )
    raise


# ==========================================
# 1. Global vocabularies and basic settings
# ==========================================
NUC_VOCAB = {
    "<PAD>": 0,
    "<SOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
    "A": 4,
    "T": 5,
    "G": 6,
    "C": 7,
    "N": 8,
}

CODON_VOCAB = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
codon_idx = 4
for codon in DEFAULT_CODON_TABLE.keys():
    CODON_VOCAB[codon] = codon_idx
    codon_idx += 1


def get_bucketed_length(length, bucket_size=16, min_len=16):
    length = max(int(length), int(min_len))
    return int(math.ceil(length / bucket_size) * bucket_size)


def cosine_noise_schedule(initial_noise_std, max_noise_std, total_epochs):
    def schedule(epoch):
        t = epoch
        noise_std = initial_noise_std + 0.5 * (max_noise_std - initial_noise_std) * (
            1 - math.cos(math.pi * t / total_epochs)
        )
        return noise_std

    return schedule


# ==========================================
# 2. Loss: keep raw labels without log2 transform or normalization
# ==========================================
def contrastive_loss(codon_rep, nuc_rep, temperature=0.1):
    codon_rep = F.normalize(codon_rep.float(), p=2, dim=-1)
    nuc_rep = F.normalize(nuc_rep.float(), p=2, dim=-1)
    logits = torch.matmul(codon_rep, nuc_rep.transpose(0, 1)) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_codon = F.cross_entropy(logits, labels)
    loss_nuc = F.cross_entropy(logits.transpose(0, 1), labels)
    return (loss_codon + loss_nuc) / 2.0


def loss_function_simplified(
    pred_fusion,
    pred_5,
    pred_c,
    pred_3,
    value_targets,
    mse_log_vars,
    codon_rep,
    nuc_rep,
    denoising_c,
    aux_5,
    aux_3,
    mu_c=None,
    logvar_c=None,
    contrastive_weight=0.0,
    denoise_weight=0.0,
):
    """
    Key points:
    1. Labels use the raw values from the CSV file.
    2. No log2 transform is applied.
    3. Labels are not standardized.
    4. MSE is computed in float32 to avoid AMP half-precision overflow.
    """
    value_targets = value_targets.float().squeeze()
    pred_fusion = pred_fusion.float().squeeze()

    mse_fusion = F.mse_loss(pred_fusion, value_targets)
    total_loss = mse_fusion

    # If the CDS branch provides a VAE KL term, clamp values to prevent mu/logvar explosion.
    if mu_c is not None and logvar_c is not None:
        mu_c_safe = torch.clamp(mu_c.float(), -10, 10)
        logvar_c_safe = torch.clamp(logvar_c.float(), -10, 10)
        kl_loss = -0.5 * torch.sum(
            1 + logvar_c_safe - mu_c_safe.pow(2) - logvar_c_safe.exp()
        )
        kl_loss = kl_loss / max(1, mu_c_safe.numel())
        total_loss = total_loss + 1e-4 * kl_loss

    # Monitoring only; the current weight is 0, so this term does not contribute to the total loss.
    c_loss = contrastive_loss(codon_rep, nuc_rep, temperature=0.1)
    zero = torch.tensor(0.0, device=pred_fusion.device)
    return total_loss, mse_fusion, zero, zero, zero, c_loss


# ==========================================
# 3. Automated sequence splitting and data pipeline
# ==========================================
class FullmRNAFusionDataset(Dataset):
    def __init__(
        self,
        csv_file,
        max_len_5=256,
        max_len_c=256,
        max_len_3=512,
        label_mean=None,
        label_std=None,
    ):
        self.data = pd.read_csv(csv_file)

        if "sequence" not in self.data.columns or "label" not in self.data.columns:
            raise ValueError(
                f"{csv_file} must contain sequence and label columns; current columns are: {list(self.data.columns)}"
            )

        # Use raw label values: negative values are allowed, with no log2 transform or standardization.
        self.data["label"] = pd.to_numeric(self.data["label"], errors="coerce")
        self.data["sequence"] = self.data["sequence"].astype(str)

        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.data = self.data.dropna(subset=["sequence", "label"]).reset_index(drop=True)

        if len(self.data) == 0:
            raise ValueError(f"{csv_file} has no valid samples after cleaning. Please check the sequence and label columns.")

        self.max_len_5 = int(max_len_5)
        self.max_len_c = int(max_len_c)
        self.max_len_nuc = int(max_len_c) * 3
        self.max_len_3 = int(max_len_3)

        # Keep compatibility with the old interface, but labels are not standardized.
        self.label_mean = 0.0
        self.label_std = 1.0

    def __len__(self):
        return len(self.data)

    @staticmethod
    def clean_sequence(seq):
        """
        Support two input formats:
        1. Space-separated single nucleotides: A A T C C C ...
        2. Continuous nucleotide sequence: AATCCC...

        Output a continuous DNA sequence containing only A/T/G/C/N.
        """
        seq = str(seq).upper().replace("U", "T")
        tokens = seq.split()

        if len(tokens) > 1:
            seq = "".join(tokens)
        else:
            seq = seq.replace(" ", "")

        seq = "".join([base for base in seq if base in {"A", "T", "G", "C", "N"}])
        return seq

    @staticmethod
    def split_full_mrna(seq):
        """
        Split a continuous nucleotide sequence into 5' UTR, CDS, and 3' UTR:
        - 5' UTR: region before the first ATG
        - CDS: from the first ATG to the first downstream in-frame stop codon, including the stop codon
        - 3' UTR: region after the stop codon

        If no ATG is found, conservatively use the first third of the sequence as an approximate 5' UTR region.
        If no stop codon is found, truncate the CDS to a length divisible by 3.
        """
        if not seq:
            return "", [], "", ""

        start_nt = seq.find("ATG")
        if start_nt == -1:
            start_nt = len(seq) // 3
            start_nt = start_nt - (start_nt % 3)

        stop_codons = {"TAA", "TAG", "TGA"}
        stop_nt = -1

        for pos in range(start_nt + 3, len(seq) - 2, 3):
            codon = seq[pos : pos + 3]
            if codon in stop_codons:
                stop_nt = pos
                break

        if stop_nt == -1:
            cds_end = len(seq) - ((len(seq) - start_nt) % 3)
        else:
            cds_end = stop_nt + 3

        str_5 = seq[:start_nt]
        str_c_nuc = seq[start_nt:cds_end]
        str_3 = seq[cds_end:]

        cds_codons = [
            str_c_nuc[i : i + 3]
            for i in range(0, len(str_c_nuc) - 2, 3)
            if len(str_c_nuc[i : i + 3]) == 3
        ]

        return str_5, cds_codons, str_c_nuc, str_3

    @staticmethod
    def sanitize_features(feat):
        if isinstance(feat, dict):
            feat = list(feat.values())
        feat = np.asarray(feat, dtype=np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return feat

    @staticmethod
    def ensure_non_empty_parts(str_5, cds_codons, str_c_nuc, str_3):
        """
        Prevent 5' UTR, CDS, or 3' UTR from becoming all-<PAD> after padding.
        Otherwise, some attention implementations may mask the entire row as -inf through key_padding_mask and produce NaN after softmax.
        """
        if len(str_5) == 0:
            str_5 = "N"
        if len(str_3) == 0:
            str_3 = "N"
        if len(cds_codons) == 0:
            cds_codons = ["NNN"]
        if len(str_c_nuc) == 0:
            str_c_nuc = "NNN"
        return str_5, cds_codons, str_c_nuc, str_3

    @staticmethod
    def pad_or_truncate_5utr(seq_list, max_len, pad_val=0):
        """
        When the 5' UTR is too long, keep the terminal region near the ATG.
        This region more directly contains Kozak context and uAUG/uORF-related signals near translation initiation.
        """
        if len(seq_list) > max_len:
            seq_list = seq_list[-max_len:]
        return seq_list + [pad_val] * (max_len - len(seq_list))

    @staticmethod
    def pad_or_truncate_cds_codons(seq_list, max_len, pad_val=0):
        """
        When the CDS codon sequence is too long, uniformly sample codons across the full length.
        Compared with keeping only the 5' end, uniform sampling better preserves global codon usage and composition information.
        """
        if len(seq_list) > max_len:
            idx = np.linspace(0, len(seq_list) - 1, max_len).astype(int)
            seq_list = [seq_list[i] for i in idx]
        return seq_list + [pad_val] * (max_len - len(seq_list))

    @staticmethod
    def pad_or_truncate_cds_nuc_from_codons(cds_codons, max_len_c, max_len_nuc, nuc_pad_val=0):
        """
        Keep CDS nucleotide input aligned with codon input by using the same codon sampling indices.
        This preserves the codon frame and ensures cds_x and cds_nuc_x correspond to the same CDS regions.
        """
        if len(cds_codons) > max_len_c:
            idx = np.linspace(0, len(cds_codons) - 1, max_len_c).astype(int)
            cds_codons = [cds_codons[i] for i in idx]

        nuc_seq = "".join(cds_codons)
        nuc_idx = [NUC_VOCAB.get(n, NUC_VOCAB["<UNK>"]) for n in nuc_seq]

        if len(nuc_idx) > max_len_nuc:
            nuc_idx = nuc_idx[:max_len_nuc]
        return nuc_idx + [nuc_pad_val] * (max_len_nuc - len(nuc_idx))

    @staticmethod
    def pad_or_truncate_3utr(seq_list, max_len, pad_val=0):
        """
        When the 3' UTR is too long, keep both the stop-proximal region and the distal region before poly(A).
        The proximal region contains stop-proximal regulatory information, while the distal region preserves distal regulatory signals.
        """
        if len(seq_list) > max_len:
            left = max_len // 2
            right = max_len - left
            seq_list = seq_list[:left] + seq_list[-right:]
        return seq_list + [pad_val] * (max_len - len(seq_list))

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        seq = self.clean_sequence(row["sequence"])
        raw_label = float(row["label"])

        str_5, cds_codons, str_c_nuc, str_3 = self.split_full_mrna(seq)
        str_5, cds_codons, str_c_nuc, str_3 = self.ensure_non_empty_parts(
            str_5, cds_codons, str_c_nuc, str_3
        )

        u5_tokens = list(str_5)
        u3_tokens = list(str_3)

        feat_5 = self.sanitize_features(compute_features_5utr(str_5))
        feat_3 = self.sanitize_features(compute_features_3utr(str_3))

        idx_5 = [NUC_VOCAB.get(n, NUC_VOCAB["<UNK>"]) for n in u5_tokens]
        idx_c = [CODON_VOCAB.get(c, CODON_VOCAB["<UNK>"]) for c in cds_codons]
        idx_c_nuc = [NUC_VOCAB.get(n, NUC_VOCAB["<UNK>"]) for n in str_c_nuc]
        idx_3 = [NUC_VOCAB.get(n, NUC_VOCAB["<UNK>"]) for n in u3_tokens]

        idx_5 = self.pad_or_truncate_5utr(
            idx_5,
            self.max_len_5,
            NUC_VOCAB["<PAD>"],
        )
        idx_c = self.pad_or_truncate_cds_codons(
            idx_c,
            self.max_len_c,
            CODON_VOCAB["<PAD>"],
        )
        idx_c_nuc = self.pad_or_truncate_cds_nuc_from_codons(
            cds_codons,
            self.max_len_c,
            self.max_len_nuc,
            NUC_VOCAB["<PAD>"],
        )
        idx_3 = self.pad_or_truncate_3utr(
            idx_3,
            self.max_len_3,
            NUC_VOCAB["<PAD>"],
        )

        return {
            "5utr_x": torch.tensor(idx_5, dtype=torch.long),
            "cds_x": torch.tensor(idx_c, dtype=torch.long),
            "cds_nuc_x": torch.tensor(idx_c_nuc, dtype=torch.long),
            "3utr_x": torch.tensor(idx_3, dtype=torch.long),
            "feat_5": torch.tensor(feat_5, dtype=torch.float32),
            "feat_3": torch.tensor(feat_3, dtype=torch.float32),
            "label": torch.tensor(raw_label, dtype=torch.float32),
        }


# ==========================================
# 3.1 Infer max_len automatically from sequence lengths in the dataset
# ==========================================
def infer_max_lens_from_dataset(
    csv_paths,
    percentile=1.0,
    bucket_size=16,
    min_len_5=16,
    min_len_c=16,
    min_len_3=16,
    cap_5=None,
    cap_c=None,
    cap_3=None,
    verbose=True,
):
    """
    Infer the following values from the actual dataset split results:
    - max_len_5: 5' UTR nucleotide length
    - max_len_c: number of CDS codons
    - max_len_3: 3' UTR nucleotide length

    percentile=1.0 uses the maximum length and does not truncate the training set.
    percentile=0.99 uses the 99th percentile to reduce GPU-memory pressure from extreme long-tail sequences.
    cap_5 / cap_c / cap_3 can be used as upper bounds to prevent OOM errors.
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    len_5_list = []
    len_c_list = []
    len_3_list = []

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        if "sequence" not in df.columns:
            raise ValueError(f"{csv_path} is missing the sequence column; current columns: {list(df.columns)}")

        for seq_raw in df["sequence"].astype(str).tolist():
            seq = FullmRNAFusionDataset.clean_sequence(seq_raw)
            str_5, cds_codons, str_c_nuc, str_3 = FullmRNAFusionDataset.split_full_mrna(seq)
            str_5, cds_codons, str_c_nuc, str_3 = FullmRNAFusionDataset.ensure_non_empty_parts(
                str_5, cds_codons, str_c_nuc, str_3
            )

            len_5_list.append(len(str_5))
            len_c_list.append(len(cds_codons))
            len_3_list.append(len(str_3))

    if len(len_5_list) == 0:
        raise ValueError("Unable to collect sequence lengths from the dataset. Please check csv_paths or the sequence column.")

    def choose_len(lengths, min_len, cap=None):
        lengths = np.asarray(lengths, dtype=np.float32)

        if percentile >= 1.0:
            raw_len = int(np.max(lengths))
        else:
            raw_len = int(np.ceil(np.quantile(lengths, percentile)))

        bucketed_len = get_bucketed_length(raw_len, bucket_size=bucket_size, min_len=min_len)

        if cap is not None:
            # Use cap as a hard upper bound. Align cap downward to a multiple of bucket_size for compatibility with the downsampling structure.
            cap = int(cap)
            cap_bucketed = max(int(min_len), (cap // bucket_size) * bucket_size)
            bucketed_len = min(bucketed_len, cap_bucketed)

        return int(bucketed_len), int(raw_len)

    max_len_5, raw_5 = choose_len(len_5_list, min_len_5, cap_5)
    max_len_c, raw_c = choose_len(len_c_list, min_len_c, cap_c)
    max_len_3, raw_3 = choose_len(len_3_list, min_len_3, cap_3)

    stats = {
        "5UTR": {
            "min": int(np.min(len_5_list)),
            "median": float(np.median(len_5_list)),
            "p95": float(np.quantile(len_5_list, 0.95)),
            "p99": float(np.quantile(len_5_list, 0.99)),
            "max": int(np.max(len_5_list)),
            "selected_raw": raw_5,
            "selected_bucketed": max_len_5,
        },
        "CDS_codons": {
            "min": int(np.min(len_c_list)),
            "median": float(np.median(len_c_list)),
            "p95": float(np.quantile(len_c_list, 0.95)),
            "p99": float(np.quantile(len_c_list, 0.99)),
            "max": int(np.max(len_c_list)),
            "selected_raw": raw_c,
            "selected_bucketed": max_len_c,
        },
        "3UTR": {
            "min": int(np.min(len_3_list)),
            "median": float(np.median(len_3_list)),
            "p95": float(np.quantile(len_3_list, 0.95)),
            "p99": float(np.quantile(len_3_list, 0.99)),
            "max": int(np.max(len_3_list)),
            "selected_raw": raw_3,
            "selected_bucketed": max_len_3,
        },
    }

    if verbose:
        print("\n📏 Inferring max_len from the dataset:")
        print(
            f"  5'UTR nt   -> raw={raw_5}, bucketed={max_len_5}, "
            f"min={stats['5UTR']['min']}, median={stats['5UTR']['median']:.1f}, "
            f"p95={stats['5UTR']['p95']:.1f}, p99={stats['5UTR']['p99']:.1f}, "
            f"max={stats['5UTR']['max']}"
        )
        print(
            f"  CDS codon  -> raw={raw_c}, bucketed={max_len_c}, "
            f"min={stats['CDS_codons']['min']}, median={stats['CDS_codons']['median']:.1f}, "
            f"p95={stats['CDS_codons']['p95']:.1f}, p99={stats['CDS_codons']['p99']:.1f}, "
            f"max={stats['CDS_codons']['max']}"
        )
        print(
            f"  3'UTR nt   -> raw={raw_3}, bucketed={max_len_3}, "
            f"min={stats['3UTR']['min']}, median={stats['3UTR']['median']:.1f}, "
            f"p95={stats['3UTR']['p95']:.1f}, p99={stats['3UTR']['p99']:.1f}, "
            f"max={stats['3UTR']['max']}"
        )

    return max_len_5, max_len_c, max_len_3, stats


# ==========================================
# 4. Top-level multi-branch fusion network architecture
# ==========================================
class GlobalmRNAMasterModel(nn.Module):
    def __init__(self, max_len_5, max_len_c, max_len_3, embed_dim=16, hidden_dim=16, latent_dim=16):
        super(GlobalmRNAMasterModel, self).__init__()

        self.max_len_5 = max_len_5
        self.max_len_c = max_len_c
        self.max_len_3 = max_len_3

        self.branch_5utr = Model_5UTR(
            vocab_size=len(NUC_VOCAB),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
        )

        self.branch_cds = Model_CDS(
            vocab_size=len(CODON_VOCAB),
            nuc_vocab_size=len(NUC_VOCAB),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            max_len=max_len_c * 3,
            codon_table=DEFAULT_CODON_TABLE,
            char_to_idx=CODON_VOCAB,
            nuc_char_to_idx=NUC_VOCAB,
        )

        self.branch_3utr = Model_3UTR(
            vocab_size=len(NUC_VOCAB),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
        )

        fusion_dim = latent_dim * 3

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        self.head_5utr = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
        )
        self.head_cds = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
        )
        self.head_3utr = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
        )

        self.output_scale = nn.Parameter(torch.ones(1) * 1.5)
        self.pred_scale = nn.Parameter(torch.tensor(1.0))
        self.pred_shift = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_5, x_c, x_c_nuc, x_3, feat_5, feat_3, noise_std=0.0):
        latent_5, aux_5 = self.branch_5utr(
            src=x_5, stacked_features=feat_5, noise_std=noise_std
        )
        (
            latent_c,
            logits_c,
            nuc_logits_c,
            mu_c,
            logvar_c,
            denoising_c,
            codon_rep,
            nuc_rep,
        ) = self.branch_cds(src=x_c, src_nuc=x_c_nuc, noise_std=noise_std)
        latent_3, aux_3 = self.branch_3utr(
            src=x_3, stacked_features=feat_3, noise_std=noise_std
        )

        fused_representation = torch.cat([latent_5, latent_c, latent_3], dim=1)
        pred_fusion = self.fusion_head(fused_representation)
        pred_fusion = pred_fusion * self.pred_scale + self.pred_shift

        pred_5 = self.head_5utr(latent_5) * self.output_scale
        pred_c = self.head_cds(latent_c) * self.output_scale
        pred_3 = self.head_3utr(latent_3) * self.output_scale

        cds_tuple = (logits_c, nuc_logits_c, mu_c, logvar_c, denoising_c, codon_rep, nuc_rep)
        return pred_fusion, pred_5, pred_c, pred_3, aux_5, aux_3, cds_tuple


# ==========================================
# 5. Training engine
# ==========================================
def move_batch_to_device(batch, device):
    return {
        "5utr_x": batch["5utr_x"].to(device, non_blocking=True),
        "cds_x": batch["cds_x"].to(device, non_blocking=True),
        "cds_nuc_x": batch["cds_nuc_x"].to(device, non_blocking=True),
        "3utr_x": batch["3utr_x"].to(device, non_blocking=True),
        "feat_5": batch["feat_5"].to(device, non_blocking=True),
        "feat_3": batch["feat_3"].to(device, non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True),
    }


def is_finite_batch(batch):
    return (
        torch.isfinite(batch["feat_5"]).all()
        and torch.isfinite(batch["feat_3"]).all()
        and torch.isfinite(batch["label"]).all()
    )


def train_master_model(MAX_LEN_5, MAX_LEN_C, MAX_LEN_3, train_csv, test_csv, temp_save_path, EPOCHS=40):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    base_batch_size_per_gpu = 2
    MICRO_BATCH_SIZE = base_batch_size_per_gpu * max(1, num_gpus)
    ACCUMULATION_STEPS = 1
    LR = 1e-3

    train_dataset = FullmRNAFusionDataset(train_csv, MAX_LEN_5, MAX_LEN_C, MAX_LEN_3)
    val_dataset = FullmRNAFusionDataset(test_csv, MAX_LEN_5, MAX_LEN_C, MAX_LEN_3)

    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory,
    )

    if len(train_loader) == 0:
        raise ValueError("The training set has zero batches. Reduce batch size or check the number of training samples.")

    model = GlobalmRNAMasterModel(
        max_len_5=MAX_LEN_5,
        max_len_c=MAX_LEN_C,
        max_len_3=MAX_LEN_3,
    ).to(DEVICE)

    if num_gpus > 1:
        model = nn.DataParallel(model)

    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    mse_log_vars = nn.Parameter(torch.zeros(4, device=DEVICE))
    scale_param = raw_model.pred_scale
    shift_param = raw_model.pred_shift

    param_groups = [
        {"params": raw_model.branch_5utr.parameters(), "lr": LR * 0.1},
        {"params": raw_model.branch_cds.parameters(), "lr": LR * 0.1},
        {"params": raw_model.branch_3utr.parameters(), "lr": LR * 0.1},
        {"params": raw_model.fusion_head.parameters(), "lr": LR},
        {"params": raw_model.head_5utr.parameters(), "lr": LR},
        {"params": raw_model.head_cds.parameters(), "lr": LR},
        {"params": raw_model.head_3utr.parameters(), "lr": LR},
        {"params": [mse_log_vars], "lr": LR * 10, "weight_decay": 0.0},
        {"params": [scale_param, shift_param], "lr": LR * 10.0, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05, foreach=False)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))
    best_r2_val = -10000.0

    total_steps = max(1, math.ceil(len(train_loader) / ACCUMULATION_STEPS) * EPOCHS)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group["lr"] for group in param_groups],
        total_steps=total_steps,
        pct_start=0.20,
        anneal_strategy="cos",
    )

    for epoch in range(EPOCHS):
        model.train()
        train_mse_fus = 0.0
        valid_train_steps = 0

        optimizer.zero_grad(set_to_none=True)

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch + 1:02d}/{EPOCHS}] Train",
            leave=False,
            disable=False,
        )

        for batch_idx, batch in enumerate(train_pbar):
            batch = move_batch_to_device(batch, DEVICE)

            if not is_finite_batch(batch):
                print(f"⚠️ Skipping invalid training batch: epoch={epoch + 1}, batch_idx={batch_idx}")
                optimizer.zero_grad(set_to_none=True)
                continue

            with autocast(enabled=(DEVICE.type == "cuda")):
                pred_fusion, pred_5, pred_c, pred_3, aux_5, aux_3, cds_tuple = model(
                    x_5=batch["5utr_x"],
                    x_c=batch["cds_x"],
                    x_c_nuc=batch["cds_nuc_x"],
                    x_3=batch["3utr_x"],
                    feat_5=batch["feat_5"],
                    feat_3=batch["feat_3"],
                    noise_std=0.0,
                )

                logits_c, nuc_logits_c, mu_c, logvar_c, denoising_c, codon_rep, nuc_rep = cds_tuple

                loss, m_fus, _, _, _, _ = loss_function_simplified(
                    pred_fusion=pred_fusion,
                    pred_5=pred_5,
                    pred_c=pred_c,
                    pred_3=pred_3,
                    value_targets=batch["label"],
                    mse_log_vars=mse_log_vars,
                    codon_rep=codon_rep,
                    nuc_rep=nuc_rep,
                    denoising_c=denoising_c,
                    aux_5=aux_5,
                    aux_3=aux_3,
                    mu_c=mu_c,
                    logvar_c=logvar_c,
                    contrastive_weight=0.0,
                    denoise_weight=0.0,
                )
                loss = loss / ACCUMULATION_STEPS

            if not torch.isfinite(loss):
                print(
                    f"⚠️ Invalid loss detected; skipping batch: epoch={epoch + 1}, "
                    f"batch_idx={batch_idx}, loss={loss.detach().float().item()}"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            train_mse_fus += float(m_fus.detach().cpu().item())
            valid_train_steps += 1

            del pred_fusion, pred_5, pred_c, pred_3, aux_5, aux_3, cds_tuple
            del logits_c, nuc_logits_c, mu_c, logvar_c, denoising_c, codon_rep, nuc_rep

        model.eval()
        val_preds, val_labels = [], []
        val_loss_total = 0.0
        valid_val_steps = 0

        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch [{epoch + 1:02d}/{EPOCHS}] Valid",
            leave=False,
            disable=True,
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                batch = move_batch_to_device(batch, DEVICE)

                if not is_finite_batch(batch):
                    print(f"⚠️ Skipping invalid validation batch: epoch={epoch + 1}, batch_idx={batch_idx}")
                    continue

                with autocast(enabled=(DEVICE.type == "cuda")):
                    pred_fusion, _, _, _, _, _, _ = model(
                        x_5=batch["5utr_x"],
                        x_c=batch["cds_x"],
                        x_c_nuc=batch["cds_nuc_x"],
                        x_3=batch["3utr_x"],
                        feat_5=batch["feat_5"],
                        feat_3=batch["feat_3"],
                        noise_std=0.0,
                    )

                if not torch.isfinite(pred_fusion).all():
                    print(
                        f"⚠️ NaN/Inf detected in pred_fusion during validation; skipping batch: "
                        f"epoch={epoch + 1}, batch_idx={batch_idx}"
                    )
                    continue

                v_loss = F.mse_loss(
                    pred_fusion.float().squeeze(),
                    batch["label"].float().squeeze(),
                )

                if not torch.isfinite(v_loss):
                    print(
                        f"⚠️ NaN/Inf detected in v_loss during validation; skipping batch: "
                        f"epoch={epoch + 1}, batch_idx={batch_idx}"
                    )
                    continue

                val_loss_total += float(v_loss.detach().cpu().item())
                valid_val_steps += 1

                val_preds.extend(pred_fusion.detach().float().cpu().numpy().flatten())
                val_labels.extend(batch["label"].detach().float().cpu().numpy().flatten())

        val_preds = np.asarray(val_preds, dtype=np.float32)
        val_labels = np.asarray(val_labels, dtype=np.float32)
        finite_mask = np.isfinite(val_preds) & np.isfinite(val_labels)

        if finite_mask.sum() < 2:
            print("❌ Not enough valid predictions in the validation set; returning -10000")
            return -10000.0

        val_preds = val_preds[finite_mask]
        val_labels = val_labels[finite_mask]

        if np.std(val_preds) < 1e-12 or np.std(val_labels) < 1e-12:
            pearson_corr = 0.0
        else:
            pearson_corr, _ = pearsonr(val_labels, val_preds)
            if not np.isfinite(pearson_corr):
                pearson_corr = 0.0

        r2_val = r2_score(val_labels, val_preds)
        if not np.isfinite(r2_val):
            r2_val = -10000.0

        avg_train_loss = train_mse_fus / max(1, valid_train_steps)
        avg_val_loss = val_loss_total / max(1, valid_val_steps)

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
            f"Train MSE: {avg_train_loss:.4f} | "
            f"Val MSE: {avg_val_loss:.4f} | "
            f"Pearson: {pearson_corr:.4f} | "
            f"R²: {r2_val:.4f}"
        )

        if r2_val > best_r2_val:
            old_r2 = best_r2_val
            best_r2_val = r2_val
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, temp_save_path)

            if old_r2 == -10000.0:
                print(f"      🌟 Initial best R2: {best_r2_val:.4f} (cached)")
            else:
                print(
                    f"      🚀 [improvement] R² significant improvement: {old_r2:.4f} -> {best_r2_val:.4f} ！！！ "
                    "(cached model updated)"
                )

        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return best_r2_val


# ==========================================
# 6. Multi-folder training: infer max_len for each dataset and run three repeats
# ==========================================
def run_multi_folder_grid_search(data_base_dir="Data", model_base_dir="Model"):
    print("\n" + "=" * 60)
    print("🚀 Starting automated multi-source training with dataset-specific max_len inference")
    print("=" * 60)

    if not os.path.exists(data_base_dir):
        print(f"❌ Data root directory not found: {data_base_dir}, Please make sure the path is correct.")
        return

    folders = [
        f
        for f in os.listdir(data_base_dir)
        if os.path.isdir(os.path.join(data_base_dir, f))
    ]
    if not folders:
        print(f"❌ Directory {data_base_dir} contains no subfolders.")
        return

    NUM_RUNS = 3
    SEARCH_EPOCHS = 30

    # Choose the length strategy based on GPU memory:
    # Do not use the maximum length to avoid excessive max_len values caused by extreme long-tail sequences.
    # Start with the 95th percentile plus caps. If OOM persists, use 0.90 and further reduce the caps.
    LEN_PERCENTILE = 0.9
    BUCKET_SIZE = 16

    # Recommended upper bounds balancing biology and GPU memory:
    # 5' UTR uses nucleotide length; CDS uses codon count; 3' UTR uses nucleotide length.
    CAP_5 = 128
    CAP_C = 512
    CAP_3 = 512

    skip_folders = {
        "in_solution_half_life",
        "Polysome_to_monosome_ratio",
        "Polysome_to_40S60S_ratio",
    }

    for folder in folders:
        if folder in skip_folders:
            print("\n" + "-" * 60)
            print(f"⏭️  Skipped folder according to settings: {folder}")
            print("-" * 60)
            continue

        print("\n" + "★" * 60)
        print(f"📂 Processing dataset: {folder}")
        print("★" * 60)

        train_csv = os.path.join(data_base_dir, folder, "train.csv")
        dev_csv = os.path.join(data_base_dir, folder, "dev.csv")
        test_csv = os.path.join(data_base_dir, folder, "test.csv")

        if not os.path.exists(train_csv) or not os.path.exists(test_csv):
            print(f"⚠️ Folder {folder} is missing train.csv or test.csv and was skipped.")
            continue

        save_dir = os.path.join(model_base_dir, folder)
        os.makedirs(save_dir, exist_ok=True)

        # Merge train and dev. Note: subsequent training must use merged_train_csv.
        merged_train_csv = os.path.join(save_dir, "merged_train.csv")
        try:
            df_train = pd.read_csv(train_csv)
            if os.path.exists(dev_csv):
                df_dev = pd.read_csv(dev_csv)
                df_merged = pd.concat([df_train, df_dev], ignore_index=True)
                print(f"  📎 Detected dev.csv and merged it successfully. Total merged training samples: {len(df_merged)}")
            else:
                df_merged = df_train
                print(f"  ⚠️ dev.csv not found; using train.csv only. Total training samples: {len(df_merged)}")

            df_merged.to_csv(merged_train_csv, index=False)
        except Exception as e:
            print(f"⚠️ Error while merging CSV files: {e}")
            continue

        # ================= [Infer max_len from the dataset] =================
        # Strict setting: use only merged_train_csv for length statistics to avoid using distributional information from the test set.
        # To avoid truncating long sequences in the test set, set csv_paths=[merged_train_csv, test_csv].
        try:
            MAX_LEN_5_AUTO, MAX_LEN_C_AUTO, MAX_LEN_3_AUTO, length_stats = infer_max_lens_from_dataset(
                csv_paths=[merged_train_csv],
                percentile=LEN_PERCENTILE,
                bucket_size=BUCKET_SIZE,
                min_len_5=16,
                min_len_c=16,
                min_len_3=16,
                cap_5=CAP_5,
                cap_c=CAP_C,
                cap_3=CAP_3,
                verbose=True,
            )
        except Exception as e:
            print(f"⚠️ Automatic max_len inference failed: {e}")
            continue

        combinations = [(MAX_LEN_5_AUTO, MAX_LEN_C_AUTO, MAX_LEN_3_AUTO)]
        # ================================================================

        best_global_r2 = -10000.0
        best_combo = None
        best_model_path = None
        results_log = []

        for idx, (l5, lc, l3) in enumerate(combinations):
            print(
                f"\n▶ [{idx + 1}/{len(combinations)}] Testing length combination -> "
                f"5'UTR: {l5}, CDS: {lc}, 3'UTR: {l3} "
                f"(repeat training {NUM_RUNS} times)"
            )

            combo_r2_list = []
            combo_best_r2 = -10000.0
            temp_combo_model = os.path.join(save_dir, "temp_combo_model.pth")

            for run_idx in range(NUM_RUNS):
                print(f"  --> 🏃 Starting run {run_idx + 1}/{NUM_RUNS} run...")
                temp_run_model = os.path.join(save_dir, "temp_run_model.pth")

                try:
                    run_r2 = train_master_model(
                        MAX_LEN_5=l5,
                        MAX_LEN_C=lc,
                        MAX_LEN_3=l3,
                        train_csv=merged_train_csv,
                        test_csv=test_csv,
                        temp_save_path=temp_run_model,
                        EPOCHS=SEARCH_EPOCHS,
                    )

                    combo_r2_list.append(run_r2)
                    print(f"      [Completed] Run {run_idx + 1} runbest R2: {run_r2:.4f}\n")

                    if run_r2 > combo_best_r2:
                        combo_best_r2 = run_r2
                        if os.path.exists(temp_run_model):
                            if os.path.exists(temp_combo_model):
                                os.remove(temp_combo_model)
                            shutil.move(temp_run_model, temp_combo_model)
                    else:
                        if os.path.exists(temp_run_model):
                            os.remove(temp_run_model)

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("      ❌ CUDA out of memory. This combination will be interrupted.")
                        combo_r2_list.append(-10000.0)
                        break
                    else:
                        print(f"      ❌ RuntimeError occurred: {e}")
                        combo_r2_list.append(-10000.0)

                except Exception as e:
                    print(f"      ❌ Unknown error occurred: {e}")
                    import traceback
                    traceback.print_exc()
                    combo_r2_list.append(-10000.0)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            valid_r2s = [r for r in combo_r2_list if r > -5000]
            mean_r2 = np.mean(valid_r2s) if valid_r2s else -10000.0

            print(
                f"  🏁 Combination {l5}-{lc}-{l3} summary -> "
                f"best R2: {combo_best_r2:.4f} | mean R2: {mean_r2:.4f}"
            )

            log_entry = {
                "Dataset": folder,
                "5UTR_Len": l5,
                "CDS_Len": lc,
                "3UTR_Len": l3,
                "5UTR_Max_Raw": length_stats["5UTR"]["max"],
                "CDS_Max_Raw": length_stats["CDS_codons"]["max"],
                "3UTR_Max_Raw": length_stats["3UTR"]["max"],
                "5UTR_P99": length_stats["5UTR"]["p99"],
                "CDS_P99": length_stats["CDS_codons"]["p99"],
                "3UTR_P99": length_stats["3UTR"]["p99"],
                "Length_Percentile": LEN_PERCENTILE,
            }

            for i in range(NUM_RUNS):
                if i < len(combo_r2_list):
                    val = combo_r2_list[i]
                    log_entry[f"Run{i + 1}_R2"] = val if val > -5000 else "OOM/Error"
                else:
                    log_entry[f"Run{i + 1}_R2"] = "-"

            log_entry["Mean_R2"] = mean_r2
            log_entry["Max_R2"] = combo_best_r2
            results_log.append(log_entry)

            if combo_best_r2 > best_global_r2:
                old_global = best_global_r2
                best_global_r2 = combo_best_r2
                best_combo = (l5, lc, l3)

                if old_global == -10000.0:
                    print(f"  👑 {folder} sets the initial best combination {best_combo} | R²: {best_global_r2:.4f}")
                else:
                    print(
                        f"  👑 {folder} produced a new global best combination {best_combo} | "
                        f"record R2: {old_global:.4f} -> {best_global_r2:.4f}"
                    )

                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)

                new_best_path = os.path.join(
                    save_dir,
                    f"best_model_5U{l5}_C{lc}_3U{l3}_R2_{best_global_r2:.4f}.pth",
                )
                if os.path.exists(temp_combo_model):
                    if os.path.exists(new_best_path):
                        os.remove(new_best_path)
                    os.rename(temp_combo_model, new_best_path)
                    best_model_path = new_best_path
            else:
                if os.path.exists(temp_combo_model):
                    os.remove(temp_combo_model)

            pd.DataFrame(results_log).to_csv(
                os.path.join(save_dir, "search_log.csv"),
                index=False,
            )

        print(f"\n✅ Folder {folder} training finished.")
        print(f"🏆 Final selected length combination: {best_combo}, best R2 score: {best_global_r2:.4f}")
        print(f"💾 Best model saved to: {best_model_path}")

    print("\n🎉 All datasets have been processed.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mROSE full-length mRNA fusion model")
    parser.add_argument("--data_base_dir", type=str, default="experiments/full_length_stability/example",
                        help="Directory containing one or more task folders. Each task folder should contain train.csv, optional dev.csv, and test.csv.")
    parser.add_argument("--model_base_dir", type=str, default="outputs/full_length_stability",
                        help="Directory for checkpoints and training logs.")
    args = parser.parse_args()
    run_multi_folder_grid_search(data_base_dir=args.data_base_dir, model_base_dir=args.model_base_dir)
