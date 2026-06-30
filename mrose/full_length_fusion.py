import os
import argparse
import math
import warnings
import gc

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
class RegionalCrossAttentionBlock(nn.Module):
    """Use one region as query and the other two regions as context."""

    def __init__(self, latent_dim, num_heads=4, dropout=0.2, ffn_multiplier=2):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.query_norm = nn.LayerNorm(latent_dim)
        self.context_norm = nn.LayerNorm(latent_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(latent_dim)

        ffn_dim = latent_dim * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, latent_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(self, query_region, context_regions, return_attention=False):
        if query_region.dim() != 2:
            raise ValueError(
                f"query_region must have shape [B, D], got {tuple(query_region.shape)}"
            )
        if context_regions.dim() != 3 or context_regions.size(1) != 2:
            raise ValueError(
                "context_regions must have shape [B, 2, D], "
                f"got {tuple(context_regions.shape)}"
            )

        query = self.query_norm(query_region).unsqueeze(1)
        context = self.context_norm(context_regions)
        attention_output, attention_weights = self.cross_attention(
            query=query,
            key=context,
            value=context,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        enhanced = self.attention_norm(
            query_region.unsqueeze(1) + self.attention_dropout(attention_output)
        )
        enhanced = self.output_norm(enhanced + self.ffn(enhanced)).squeeze(1)

        if return_attention:
            return enhanced, attention_weights.squeeze(2)
        return enhanced


class CrossAttentionFusion(nn.Module):
    """Fuse 5'UTR, CDS, and 3'UTR representations with three-way attention."""

    def __init__(self, latent_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.align_5utr = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.align_cds = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.align_3utr = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )

        self.attend_5utr = RegionalCrossAttentionBlock(
            latent_dim, num_heads=num_heads, dropout=dropout
        )
        self.attend_cds = RegionalCrossAttentionBlock(
            latent_dim, num_heads=num_heads, dropout=dropout
        )
        self.attend_3utr = RegionalCrossAttentionBlock(
            latent_dim, num_heads=num_heads, dropout=dropout
        )

        fusion_dim = latent_dim * 3
        self.region_fusion = nn.Sequential(
            nn.Linear(fusion_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.region_gate = nn.Sequential(
            nn.Linear(fusion_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 3),
        )
        self.output_norm = nn.LayerNorm(latent_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, latent_5, latent_c, latent_3, return_attention=False):
        for name, latent in (
            ("latent_5", latent_5),
            ("latent_c", latent_c),
            ("latent_3", latent_3),
        ):
            if latent.dim() != 2:
                raise ValueError(
                    f"{name} must have shape [B, D], got {tuple(latent.shape)}"
                )

        h5 = self.align_5utr(latent_5)
        hc = self.align_cds(latent_c)
        h3 = self.align_3utr(latent_3)
        context_for_5 = torch.stack([hc, h3], dim=1)
        context_for_c = torch.stack([h5, h3], dim=1)
        context_for_3 = torch.stack([h5, hc], dim=1)

        if return_attention:
            enhanced_5, attention_5 = self.attend_5utr(
                h5, context_for_5, return_attention=True
            )
            enhanced_c, attention_c = self.attend_cds(
                hc, context_for_c, return_attention=True
            )
            enhanced_3, attention_3 = self.attend_3utr(
                h3, context_for_3, return_attention=True
            )
        else:
            enhanced_5 = self.attend_5utr(h5, context_for_5)
            enhanced_c = self.attend_cds(hc, context_for_c)
            enhanced_3 = self.attend_3utr(h3, context_for_3)

        enhanced_regions = torch.stack(
            [enhanced_5, enhanced_c, enhanced_3], dim=1
        )
        concatenated = enhanced_regions.reshape(enhanced_regions.size(0), -1)
        gate_weights = torch.softmax(self.region_gate(concatenated), dim=-1)
        gated_representation = torch.sum(
            enhanced_regions * gate_weights.unsqueeze(-1), dim=1
        )
        fused_interaction = self.region_fusion(concatenated)
        h_mrna = self.output_norm(
            gated_representation + self.output_dropout(fused_interaction)
        )

        if return_attention:
            return h_mrna, {
                "region_gate_weights": gate_weights,
                "attention_5utr_to_cds_3utr": attention_5,
                "attention_cds_to_5utr_3utr": attention_c,
                "attention_3utr_to_5utr_cds": attention_3,
                "enhanced_5utr": enhanced_5,
                "enhanced_cds": enhanced_c,
                "enhanced_3utr": enhanced_3,
            }
        return h_mrna


class GlobalmRNAMasterModel(nn.Module):
    def __init__(
        self,
        max_len_5,
        max_len_c,
        max_len_3,
        embed_dim=16,
        hidden_dim=16,
        latent_dim=16,
        fusion_num_heads=None,
        fusion_dropout=0.2,
    ):
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

        if fusion_num_heads is None:
            fusion_num_heads = next(
                heads for heads in (4, 2, 1) if latent_dim % heads == 0
            )

        self.fusion_head = CrossAttentionFusion(
            latent_dim=latent_dim,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout,
        )
        self.mrna_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(32, 1),
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

    def forward(
        self,
        x_5,
        x_c,
        x_c_nuc,
        x_3,
        feat_5,
        feat_3,
        noise_std=0.0,
        return_fusion_attention=False,
    ):
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

        if return_fusion_attention:
            fused_representation, fusion_aux = self.fusion_head(
                latent_5, latent_c, latent_3, return_attention=True
            )
        else:
            fused_representation = self.fusion_head(latent_5, latent_c, latent_3)
            fusion_aux = None

        pred_fusion = self.mrna_head(fused_representation)
        pred_fusion = pred_fusion * self.pred_scale + self.pred_shift

        pred_5 = self.head_5utr(latent_5) * self.output_scale
        pred_c = self.head_cds(latent_c) * self.output_scale
        pred_3 = self.head_3utr(latent_3) * self.output_scale

        cds_tuple = (logits_c, nuc_logits_c, mu_c, logvar_c, denoising_c, codon_rep, nuc_rep)
        if return_fusion_attention:
            return pred_fusion, pred_5, pred_c, pred_3, aux_5, aux_3, cds_tuple, fusion_aux
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


def evaluate_regression_model(model, data_loader, device):
    model.eval()
    predictions = []
    labels = []
    loss_total = 0.0
    valid_steps = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = move_batch_to_device(batch, device)
            if not is_finite_batch(batch):
                print(f"Skipping invalid evaluation batch: batch_idx={batch_idx}")
                continue

            with autocast(enabled=(device.type == "cuda")):
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
                print(f"Skipping non-finite evaluation prediction: batch_idx={batch_idx}")
                continue

            batch_loss = F.mse_loss(
                pred_fusion.float().squeeze(),
                batch["label"].float().squeeze(),
            )
            if not torch.isfinite(batch_loss):
                print(f"Skipping non-finite evaluation loss: batch_idx={batch_idx}")
                continue

            loss_total += float(batch_loss.detach().cpu().item())
            valid_steps += 1
            predictions.extend(pred_fusion.detach().float().cpu().numpy().flatten())
            labels.extend(batch["label"].detach().float().cpu().numpy().flatten())

    predictions = np.asarray(predictions, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    finite_mask = np.isfinite(predictions) & np.isfinite(labels)
    if finite_mask.sum() < 2:
        return {"mse": float("nan"), "pearson": 0.0, "r2": -10000.0}

    predictions = predictions[finite_mask]
    labels = labels[finite_mask]
    if np.std(predictions) < 1e-12 or np.std(labels) < 1e-12:
        pearson_corr = 0.0
    else:
        pearson_corr, _ = pearsonr(labels, predictions)
        if not np.isfinite(pearson_corr):
            pearson_corr = 0.0

    r2_value = r2_score(labels, predictions)
    if not np.isfinite(r2_value):
        r2_value = -10000.0

    return {
        "mse": loss_total / max(1, valid_steps),
        "pearson": float(pearson_corr),
        "r2": float(r2_value),
    }


def train_master_model(
    MAX_LEN_5,
    MAX_LEN_C,
    MAX_LEN_3,
    train_csv,
    val_csv,
    test_csv,
    final_model_path,
    EPOCHS=40,
):
    if EPOCHS <= 0 or EPOCHS % 10 != 0:
        raise ValueError("EPOCHS must be a positive multiple of 10")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    base_batch_size_per_gpu = 2
    MICRO_BATCH_SIZE = base_batch_size_per_gpu * max(1, num_gpus)
    ACCUMULATION_STEPS = 1
    LR = 1e-3

    train_dataset = FullmRNAFusionDataset(train_csv, MAX_LEN_5, MAX_LEN_C, MAX_LEN_3)
    val_dataset = (
        FullmRNAFusionDataset(val_csv, MAX_LEN_5, MAX_LEN_C, MAX_LEN_3)
        if val_csv
        else None
    )
    test_dataset = FullmRNAFusionDataset(test_csv, MAX_LEN_5, MAX_LEN_C, MAX_LEN_3)

    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=MICRO_BATCH_SIZE,
            shuffle=False,
            pin_memory=pin_memory,
        )
    test_loader = DataLoader(
        test_dataset,
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
        {"params": raw_model.mrna_head.parameters(), "lr": LR},
        {"params": raw_model.head_5utr.parameters(), "lr": LR},
        {"params": raw_model.head_cds.parameters(), "lr": LR},
        {"params": raw_model.head_3utr.parameters(), "lr": LR},
        {"params": [mse_log_vars], "lr": LR * 10, "weight_decay": 0.0},
        {"params": [scale_param, shift_param], "lr": LR * 10.0, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05, foreach=False)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    total_steps = max(1, math.ceil(len(train_loader) / ACCUMULATION_STEPS) * EPOCHS)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group["lr"] for group in param_groups],
        total_steps=total_steps,
        pct_start=0.20,
        anneal_strategy="cos",
    )

    val_metrics = None
    test_metrics = None
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

        avg_train_loss = train_mse_fus / max(1, valid_train_steps)
        if val_loader is not None:
            val_metrics = evaluate_regression_model(model, val_loader, DEVICE)
            print(
                f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
                f"Train MSE: {avg_train_loss:.4f} | "
                f"Val MSE: {val_metrics['mse']:.4f} | "
                f"Pearson: {val_metrics['pearson']:.4f} | "
                f"R²: {val_metrics['r2']:.4f}"
            )
        else:
            val_metrics = None
            print(
                f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
                f"Train MSE: {avg_train_loss:.4f}"
            )

        if (epoch + 1) % 10 == 0:
            test_metrics = evaluate_regression_model(model, test_loader, DEVICE)
            print(
                f"Test at epoch {epoch + 1:02d} | "
                f"MSE: {test_metrics['mse']:.4f} | "
                f"Pearson: {test_metrics['pearson']:.4f} | "
                f"R²: {test_metrics['r2']:.4f}"
            )

        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    state_dict = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )
    torch.save(state_dict, final_model_path)
    return {"validation": val_metrics, "test": test_metrics}


# ==========================================
# 6. Multi-folder training: infer max_len and train each dataset once
# ==========================================
def run_multi_folder_training(data_base_dir="Data", model_base_dir="Model"):
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

        # Infer lengths from the training split only.
        try:
            MAX_LEN_5_AUTO, MAX_LEN_C_AUTO, MAX_LEN_3_AUTO, length_stats = infer_max_lens_from_dataset(
                csv_paths=[train_csv],
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

        l5, lc, l3 = MAX_LEN_5_AUTO, MAX_LEN_C_AUTO, MAX_LEN_3_AUTO
        final_model_path = os.path.join(
            save_dir,
            f"model_5U{l5}_C{lc}_3U{l3}_final.pth",
        )
        print(
            f"\n▶ Training once -> 5'UTR: {l5}, CDS: {lc}, 3'UTR: {l3}"
        )

        try:
            metrics = train_master_model(
                MAX_LEN_5=l5,
                MAX_LEN_C=lc,
                MAX_LEN_3=l3,
                train_csv=train_csv,
                val_csv=dev_csv if os.path.exists(dev_csv) else None,
                test_csv=test_csv,
                final_model_path=final_model_path,
                EPOCHS=SEARCH_EPOCHS,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("      ❌ CUDA out of memory. Dataset training was interrupted.")
            else:
                print(f"      ❌ RuntimeError occurred: {e}")
            continue
        except Exception as e:
            print(f"      ❌ Unknown error occurred: {e}")
            import traceback
            traceback.print_exc()
            continue

        val_metrics = metrics["validation"] or {
            "mse": float("nan"),
            "pearson": float("nan"),
            "r2": float("nan"),
        }
        test_metrics = metrics["test"]
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
            "Final_Val_MSE": val_metrics["mse"],
            "Final_Val_Pearson": val_metrics["pearson"],
            "Final_Val_R2": val_metrics["r2"],
            "Final_Test_MSE": test_metrics["mse"],
            "Final_Test_Pearson": test_metrics["pearson"],
            "Final_Test_R2": test_metrics["r2"],
            "Checkpoint": final_model_path,
        }
        pd.DataFrame([log_entry]).to_csv(
            os.path.join(save_dir, "training_log.csv"),
            index=False,
        )

        print(f"\n✅ Folder {folder} training finished.")
        print(f"💾 Final-epoch model saved to: {final_model_path}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n🎉 All datasets have been processed.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mROSE full-length mRNA fusion model")
    parser.add_argument("--data_base_dir", type=str, default="experiments/full_length_stability/example",
                        help="Directory containing one or more task folders. Each task folder should contain train.csv, optional dev.csv, and test.csv.")
    parser.add_argument("--model_base_dir", type=str, default="outputs/full_length_stability",
                        help="Directory for checkpoints and training logs.")
    args = parser.parse_args()
    run_multi_folder_training(data_base_dir=args.data_base_dir, model_base_dir=args.model_base_dir)
