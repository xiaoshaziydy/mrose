import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.nn import MultiheadAttention, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import copy
import os
from scipy.stats import spearmanr, pearsonr
import time
import hashlib
import signal
import itertools
import csv
import random
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
import pandas as pd
from collections import Counter
from Bio.Seq import Seq

def compute_features(utr):
    utr = str(utr).upper()  # Ensure uppercase DNA
    length = len(utr)
    total_codons = length // 3
    if total_codons == 0:
        return {f'codon_{aa}': 0 for aa in ['H','K','I','M','E','F','D','G','T','A','C','Q','S','L','Y','*','P','R','V','W','N']} | \
               {'uAUG': 0, 'uORF': 0, 'CGratio': 0, 'CGperc': 0, 'utrlen_m80': length - 80, 'ATratio': 0}

    # Translate with BioPython in frame 1 and ignore trailing incomplete codons
    try:
        seq = Seq(utr[:total_codons * 3])  # Keep only complete codons
        protein = seq.translate(to_stop=False)
        aas = list(str(protein))
    except Exception:
        aas = []
        total_codons = 0

    # Count amino-acid frequencies
    aa_count = Counter(aas)
    aa_map = {
        'H': 'codon_H', 'K': 'codon_K', 'I': 'codon_I', 'M': 'codon_M', 'E': 'codon_E',
        'F': 'codon_F', 'D': 'codon_D', 'G': 'codon_G', 'T': 'codon_T', 'A': 'codon_A',
        'C': 'codon_C', 'Q': 'codon_Q', 'S': 'codon_S', 'L': 'codon_L', 'Y': 'codon_Y',
        '*': 'codon_*', 'P': 'codon_P', 'R': 'codon_R', 'V': 'codon_V', 'W': 'codon_W', 'N': 'codon_N'
    }
    aa_freq = {aa_map[aa]: aa_count.get(aa, 0) / total_codons for aa in aa_map}

    # uAUG: number of in-frame ATG codons
    uaug = sum(1 for i in range(0, total_codons * 3, 3) if utr[i:i+3] == 'ATG')

    # uORF: number of complete ORFs with at least three non-stop amino acids before a stop codon
    min_orf_length = 3
    uorf = 0
    current_length = 0
    for aa in aas:
        if aa == '*':
            if current_length >= min_orf_length:
                uorf += 1
            current_length = 0
        else:
            current_length += 1

    # Nucleotide counts
    nuc_count = Counter(utr)
    c_num, g_num = nuc_count['C'], nuc_count['G']
    a_num, t_num = nuc_count['A'], nuc_count['T']

    # CGratio
    cgratio = abs(c_num / g_num - 1) if g_num > 0 else 0

    # CGperc
    cgperc = (c_num + g_num) / length if length > 0 else 0

    # utrlen_m80
    utrlen_m80 = length - 80

    # ATratio
    atratio = abs(a_num / t_num - 1) if t_num > 0 else 0

    # Assemble feature dictionary
    result = aa_freq
    result['uAUG'] = uaug
    result['uORF'] = uorf
    result['CGratio'] = cgratio
    result['CGperc'] = cgperc
    result['utrlen_m80'] = utrlen_m80
    result['ATratio'] = atratio
    return result

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width = x.size()
        query = self.query(x).view(batch_size, -1, width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        value = self.value(x).view(batch_size, -1, width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, -1)
        out = self.gamma * out + x
        return out

class DoubleConv1DWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            SelfAttention(mid_channels),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.main(x)
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        out += residual
        out = nn.ReLU(inplace=True)(out)
        return out

class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.main(x)
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        out += residual
        out = nn.ReLU(inplace=True)(out)
        return out

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        self.upsample_mode = 'nearest' if not bilinear else 'linear'
        self.align_corners = None
        self.extra_conv = nn.Conv1d(in_channels // 2, in_channels, kernel_size=1)
        self.reduce_conv = nn.Conv1d(in_channels * 2, in_channels, kernel_size=1)
        self.conv = DoubleConv1D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.size()[2:], mode=self.upsample_mode, align_corners=self.align_corners)
        x2 = self.extra_conv(x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.reduce_conv(x)
        return self.conv(x)

class OutConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, hidden_channels=None,
                 kernel_size=1, stride=1, padding=0, activation=nn.ReLU,
                 use_batchnorm=False, dropout_prob=0.0):
        super(OutConv1D, self).__init__()
        if hidden_channels is None:
            hidden_channels = out_channels
        layers = []
        layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, stride, padding))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_channels))
        layers.append(activation())
        if dropout_prob > 0.0:
            layers.append(nn.Dropout(p=dropout_prob))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, stride, padding))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(activation())
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(p=dropout_prob))
        layers.append(nn.Conv1d(hidden_channels, out_channels, kernel_size, stride, padding))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ComplexUNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=4, base_channels=64):
        super(ComplexUNet1D, self).__init__()
        self.inc = DoubleConv1DWithAttention(in_channels, base_channels * 8)
        self.down3 = Down(base_channels * 8, base_channels * 16)
        self.down4 = Down(base_channels * 16, base_channels * 32)
        self.down5 = Down(base_channels * 32, base_channels * 64)
        self.up1 = Up(base_channels * 64, base_channels * 32)
        self.up2 = Up(base_channels * 32, base_channels * 16)
        self.up3 = Up(base_channels * 16, base_channels * 8)
        self.outc = OutConv1D(base_channels * 8, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x4 = self.down3(x1)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, num_layers, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.conv_in = nn.Conv1d(in_channels, hidden_dim, 1)
        self.conv_out = nn.Conv1d(hidden_dim, out_channels, 1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        batch_size, c, width = x.size()
        x = self.conv_in(x)
        x = x.permute(2, 0, 1)
        x = self.encoder(x)
        x = x.permute(1, 2, 0)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        x = self.conv_out(x)
        return x

class DoubleConv1DWithTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, num_heads=2, num_layers=1, hidden_dim=128, kernel_size=7):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            TransformerBlock(hidden_dim, hidden_dim, num_heads, num_layers, hidden_dim),
            nn.Conv1d(hidden_dim, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels)
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.main(x)
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        out += residual
        out = nn.LeakyReLU(negative_slope=0.01, inplace=True)(out)
        return out

class DownWithTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, num_layers=2, hidden_dim=128):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1DWithTransformer(in_channels, out_channels, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpWithTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, num_layers=2, hidden_dim=128):
        super().__init__()
        self.upsample_mode = 'nearest'
        self.align_corners = None
        self.extra_conv = nn.Conv1d(in_channels // 2, in_channels, kernel_size=1)
        self.reduce_conv = nn.Conv1d(in_channels * 2, in_channels, kernel_size=1)
        self.conv = DoubleConv1DWithTransformer(in_channels, out_channels, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.size()[2:], mode=self.upsample_mode, align_corners=self.align_corners)
        x2 = self.extra_conv(x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.reduce_conv(x)
        return self.conv(x)

class ComplexTransformerUNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=4, base_channels=96, num_heads=1, num_layers=2, hidden_dim=128):
        super(ComplexTransformerUNet1D, self).__init__()
        self.inc = DoubleConv1DWithTransformer(in_channels, base_channels * 8, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        self.down3 = DownWithTransformer(base_channels * 8, base_channels * 16, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        self.down4 = DownWithTransformer(base_channels * 16, base_channels * 32, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        self.down5 = DownWithTransformer(base_channels * 32, base_channels * 64, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        self.up1 = UpWithTransformer(base_channels * 64, base_channels * 32, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        self.up2 = UpWithTransformer(base_channels * 32, base_channels * 16, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        self.up3 = UpWithTransformer(base_channels * 16, base_channels * 8, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        self.outc = nn.Sequential(
            nn.Conv1d(base_channels * 8, hidden_dim, 1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv1d(hidden_dim, out_channels, 1)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x4 = self.down3(x1)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.commitment_cost = commitment_cost

    def forward(self, input):
        flat_input = input.view(-1, self._embedding_dim)
        dist = (flat_input.unsqueeze(1) - self.weight) ** 2
        _, encoding_indices = (-dist).max(1)
        quantized = self.weight[encoding_indices][:,-1,:]
        e_latent_loss = F.mse_loss(quantized.detach(), input)
        q_latent_loss = F.mse_loss(quantized, input.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        avg_probs = torch.mean(torch.exp(-dist), dim=1)
        perplexity = torch.exp(torch.mean(torch.log(avg_probs)))
        return quantized, loss, perplexity, encoding_indices

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

def generate_kmer_features(sequence, k, num_features):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    kmer_features = torch.zeros(len(kmers), num_features, dtype=torch.float)
    for i, kmer in enumerate(kmers):
        hash_value = int(hashlib.md5(kmer.encode()).hexdigest(), 16) % num_features
        kmer_features[i, hash_value] = 1
    for i in range(kmer_features.shape[0]):
        row_norm = torch.linalg.norm(kmer_features[i, :], ord=2)
        if row_norm > 0:
            kmer_features[i, :] /= row_norm
    return kmer_features

def rna_to_idx(rna_seq):
    return [char_to_idx[char] for char in rna_seq]

def idx_to_rna(idx_seq):
    return ''.join([idx_to_char[idx] for idx in idx_seq])

def add_special_tokens(rna_seq_list, char_to_idx):
    return [char_to_idx['<CLS>']] + rna_seq_list

def pad_or_truncate(seq, max_len):
    if len(seq) > max_len:
        return seq[:max_len]
    else:
        return seq + [char_to_idx['EOS']] * (max_len - len(seq))

def create_mask(seq, mask_prob=0.2):
    mask = np.random.rand(len(seq)) < mask_prob
    mask[0] = mask[-1] = False
    return mask

class RNADataset(Dataset):
    def __init__(self, file_name, vocab, is_train=True):
        self.rna_sequences = []
        self.mrl_values = []
        self.vocab = vocab
        self.is_train = is_train
        self.features = []
        self.feature_cols = [
            'codon_H', 'codon_K', 'codon_I', 'uAUG', 'CGratio', 'codon_M', 'codon_E',
            'codon_F', 'codon_D', 'codon_G', 'codon_T', 'codon_A', 'codon_C', 'codon_Q',
            'codon_S', 'codon_L', 'codon_Y', 'codon_P', 'codon_R', 'codon_*','codon_V', 'codon_W',
            'codon_N', 'uORF', 'CGperc', 'utrlen_m80', 'ATratio'
        ]
        with open(file_name, 'r', encoding='latin1') as file:
            reader = csv.DictReader(file)
            for row in reader:
                rna_sequence = row['utr'].strip().upper()
                mrl_value = float(row['rl'])
                self.rna_sequences.append(rna_sequence)
                self.mrl_values.append(mrl_value)
                feature_dict = compute_features(rna_sequence)
                features = [feature_dict[col] for col in self.feature_cols]
                self.features.append(features)

    def __len__(self):
        return len(self.rna_sequences)

    def __getitem__(self, idx):
        rna_seq = self.rna_sequences[idx]
        features = self.features[idx]
        kmer_features = generate_kmer_features(rna_seq, k=3, num_features=128)

        if self.is_train and random.random() < 0.1:
            mutation_idx = random.randint(0, len(rna_seq) - 1)
            mutated_base = random.choice(['A', 'C', 'G', 'T'])
            while mutated_base == rna_seq[mutation_idx]:
                mutated_base = random.choice(['A', 'C', 'G', 'T'])
            rna_seq = rna_seq[:mutation_idx] + mutated_base + rna_seq[mutation_idx+1:]

        rna_seq_list = rna_to_idx(rna_seq)
        rna_seq_with_special = add_special_tokens(rna_seq_list, char_to_idx)
        input_tensor = torch.tensor(rna_seq_with_special, dtype=torch.long)

        return input_tensor, self.mrl_values[idx], kmer_features, torch.tensor(features)

class CNNFeatureExtractor(nn.Module):
    def __init__(self, embed_dim, num_filters, filter_size):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(embed_dim, num_filters, filter_size, padding=filter_size//2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, filter_size, padding=filter_size//2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = F.avg_pool1d(x, x.size(2))
        x = x.squeeze(2)
        return x

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, max_relative_positions=128):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.rel_pos_encoding = RelativePositionalEncoding(d_model, max_relative_positions)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        q, k, v = src2, src2, src2
        rel_pos_bias = self.rel_pos_encoding(q, k)
        attn_output, attn_output_weights = self.self_attn(
            q, k, value=v, key_padding_mask=src_key_padding_mask
        )
        attn_output = self.dropout1(attn_output)
        src = src + attn_output
        src2 = self.norm2(src)
        feedforward_output = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + feedforward_output
        return src

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_relative_positions=128):
        super(RelativePositionalEncoding, self).__init__()
        self.max_relative_positions = max_relative_positions
        self.relative_position_bias_table = nn.Parameter(
            torch.randn((2 * max_relative_positions, d_model))
        )

    def compute_distance_bucket(self, relative_positions):
        num_buckets = self.max_relative_positions // 4 + 1
        bucket_indices = (torch.abs(relative_positions) / 4).float()
        num_buckets_tensor = torch.tensor(num_buckets, dtype=bucket_indices.dtype, device=bucket_indices.device)
        bucket_indices = torch.log(bucket_indices + 1e-6) / torch.log(num_buckets_tensor)
        bucket_indices = bucket_indices.clamp(0, 1) * (num_buckets - 1)
        return bucket_indices.long()

    def forward(self, q, k):
        seq_length = q.size(1)
        relative_positions = torch.arange(seq_length, device=q.device).view(1, seq_length, 1) - \
                             torch.arange(seq_length, device=q.device).view(1, 1, seq_length)
        rel_indices = relative_positions.clamp(-self.max_relative_positions + 1, self.max_relative_positions - 1) + self.max_relative_positions
        bucket_indices = self.compute_distance_bucket(rel_indices)
        rel_pos_bias = self.relative_position_bias_table[bucket_indices.view(-1)].view(seq_length, seq_length, -1)
        return rel_pos_bias

class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, max_relative_positions=128):
        super(CustomMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rel_pos_encoding = RelativePositionalEncoding(embed_dim, max_relative_positions)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        N, seq_len, embed_dim = src.size()
        queries = self.queries(src).reshape(N, seq_len, self.num_heads, self.head_dim)
        keys = self.keys(src).reshape(N, seq_len, self.num_heads, self.head_dim)
        values = self.values(src).reshape(N, seq_len, self.num_heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        rel_pos_bias = self.rel_pos_encoding(queries, keys)
        query_len, key_len, depth = rel_pos_bias.shape

        rel_pos_bias = rel_pos_bias.unsqueeze(0)
        rel_pos_bias = rel_pos_bias.unsqueeze(0)
        rel_pos_bias = rel_pos_bias.expand(N, self.num_heads, query_len, key_len, depth)
        rel_pos_bias = rel_pos_bias.permute(0, 1, 2, 3, 4)
        rel_pos_bias = rel_pos_bias.sum(dim=-1)
        energy += rel_pos_bias

        if attn_mask is not None:
            energy += attn_mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.transpose(0, 1)
            seq_len = energy.size(2)
            key_padding_mask = key_padding_mask.unsqueeze(1).expand(-1, self.num_heads, -1)
            key_padding_mask = key_padding_mask.unsqueeze(2).repeat(1, 1, seq_len, 1)
            energy_mask = torch.zeros_like(energy, device=energy.device)
            energy_mask = energy_mask.masked_fill(key_padding_mask.bool(), float('-inf'))
            energy = energy + energy_mask

        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=-1)
        attention = self.dropout(attention)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.embed_dim)
        out = self.fc_out(out)
        out = self.leakyrelu(out)
        return out, attention

class EnhancedEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.2, max_relative_positions=64):
        super(EnhancedEncoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        attn_output, attn_output_weights = self.self_attn(src2, key_padding_mask=src_key_padding_mask, attn_mask=src_mask)
        src = src + attn_output
        src2 = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + src2
        return src

class EnhancedTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(EnhancedTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer.d_model)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return self.norm(output)

class EnhancedTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.2, max_relative_positions=64):
        super(EnhancedTransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout, max_relative_positions=max_relative_positions)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.norm1(tgt)
        tgt2, self_attn_weights = self.self_attn(tgt2, key_padding_mask=None, attn_mask=None)
        tgt = tgt + tgt2
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
        tgt = tgt + tgt2
        return tgt

class EnhancedTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(EnhancedTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(decoder_layer.d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return self.norm(output)

class VAEWithTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, num_encoder_layers,
                 num_decoder_layers, num_heads, num_embeddings, commitment_cost, kmer_feature_dim):
        super(VAEWithTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.vector_quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_scale = nn.Parameter(torch.tensor(math.sqrt(embed_dim)))
        self.pos_encoder = PositionalEncoding(embed_dim)

        encoder_layer = EnhancedEncoderLayer(d_model=embed_dim, nhead=num_heads, max_relative_positions=8)
        self.transformer_encoder = EnhancedTransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, embed_dim)
        self.decoder_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_decoder = PositionalEncoding(embed_dim)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

        decoder_layer = EnhancedTransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, max_relative_positions=8)
        self.transformer_decoder = EnhancedTransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.fc_latent_to_hidden = nn.Linear(latent_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.fc_base_pred = nn.Linear(embed_dim, vocab_size)
        self.cnn_feature_extractor0 = CNNFeatureExtractor(embed_dim, num_filters=latent_dim*2, filter_size=3)
        self.cnn_feature_extractor = CNNFeatureExtractor(embed_dim, num_filters=latent_dim, filter_size=3)
        self.fc_mfe_pred = nn.Linear(latent_dim, 1)
        
        self.unet = ComplexTransformerUNet1D(self.embed_dim, self.embed_dim)
        self.kmer_fc = nn.Linear(kmer_feature_dim, self.embed_dim)
        self.fc_z_q_to_decoder = nn.Linear(latent_dim, embed_dim)
        
        self.feature_embedding = nn.Sequential(
            nn.Linear(27, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        self.decoder_feature_extractor = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(inplace=True)
        )
        self.norm1 = nn.LayerNorm(self.embed_dim)

    def encode(self, src):
        src_mask = self._generate_square_subsequent_mask(len(src)).to(src.device)
        src = self.embedding(src) * self.embed_scale
        src = self.norm1(src)
        src_key_padding_mask = (src == 0).any(dim=-1).transpose(0, 1)
        encoder_output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        encoder_output_avg_pooled = encoder_output.mean(dim=1)
        h = F.relu(self.fc1(encoder_output_avg_pooled))
        mu = self.fc21(h)
        log_var = self.fc22(h)
        return mu, log_var, encoder_output

    def decode(self, src, z):
        tgt_seq_len = src.size(1)
        h = F.relu(self.fc3(z))
        h = self.fc4(h).unsqueeze(1).expand(-1, tgt_seq_len, -1)
        emb = self.decoder_embedding(torch.full((h.size(0), tgt_seq_len), 0, dtype=torch.long).to(h.device))
        emb = emb * math.sqrt(self.decoder_embedding.embedding_dim)
        memory = self.transformer_encoder(emb)
        output = self.transformer_decoder(emb, memory)
        return output

    def forward(self, src, kmer_features, stacked_features, noise_std=0.0):
        mu, log_var, encoder_output = self.encode(src)
        z = self.reparameterize(mu, log_var)
        z_q, vq_loss, _, _ = self.vector_quantizer(z)
        noisy_encoder_output = encoder_output + noise_std * torch.randn_like(encoder_output)
        denoised_output = self.unet(noisy_encoder_output.permute(0, 2, 1))
        decoder_input = denoised_output.permute(0, 2, 1)
        decoder_input_features = self.cnn_feature_extractor(decoder_input)
        decoder_output = self.decode(src, decoder_input_features)

        logits = self.fc_out(decoder_output)
        kmer_emb = self.kmer_fc(kmer_features)
        feature_embedded = self.feature_embedding(stacked_features.unsqueeze(1))
        combined_features = torch.cat([denoised_output.permute(0, 2, 1), feature_embedded, kmer_emb], dim=1)
        cnn_features = self.cnn_feature_extractor(combined_features)
        mfe_pred = self.fc_mfe_pred(cnn_features)
        denoising_loss = F.mse_loss(denoised_output.permute(0, 2, 1), encoder_output)
        
        return logits, mu, log_var, mfe_pred, vq_loss, denoising_loss

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

def cosine_noise_schedule(initial_noise_std, max_noise_std, total_epochs):
    def schedule(epoch):
        t = epoch
        beta_start = initial_noise_std
        beta_end = max_noise_std
        T = total_epochs
        noise_std = beta_start + 0.5 * (beta_end - beta_start) * (1 - np.cos(np.pi * t / T))
        return noise_std
    return schedule

def loss_function(logits, targets, mu, logvar, vq_loss, denoising_loss, mfe_preds,
                  mfe_targets, vocab, beta=1.0, mask_weight=1.0, mfe_weight=1.0, denoise_weight=1.0, vq_beta=1.0):
    cross_entropy = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    mfe_loss = F.mse_loss(mfe_preds.squeeze().unsqueeze(0), mfe_targets.squeeze().unsqueeze(0))
    return cross_entropy + beta * KLD + mfe_weight * mfe_loss + vq_beta * vq_loss + denoise_weight * denoising_loss, mfe_loss, denoising_loss

def calculate_accuracy(predicted, targets, mask):
    _, predicted_indices = torch.max(predicted, dim=1)
    correct = (predicted_indices == targets) & mask.bool()
    accuracy = correct.float().sum() / mask.sum().item()
    return accuracy.item()

def collate_fn(batch, max_len=150, vocab=None):
    seq_tensors, mfe_values, kmer_features, features = zip(*batch)
    max_len_with_eos = max(len(seq) for seq in seq_tensors) if max_len is None else max_len
    max_len_without_eos = max_len_with_eos - 1
    max_len_kmer = max(feat.size(0) for feat in kmer_features)
    
    padded_seqs = torch.full((len(seq_tensors), max_len_with_eos), vocab['<PAD>'], dtype=torch.long)
    padded_kmer_features = []
    padded_features = []

    for i, (seq, feat, extra_feat) in enumerate(zip(seq_tensors, kmer_features, features)):
        if seq.size(0) > max_len_with_eos:
            seq = seq[:max_len_with_eos]
        padded_seqs[i, :seq.size(0)] = seq
        padded_feat = F.pad(feat, (0, 0, 0, max_len_kmer - feat.size(0)), mode='constant', value=0)
        padded_kmer_features.append(padded_feat)
        padded_features.append(extra_feat.clone().detach())

    stacked_kmer_features = torch.stack(padded_kmer_features)
    stacked_features = torch.stack(padded_features)
    
    return padded_seqs, torch.tensor(mfe_values, dtype=torch.float), stacked_kmer_features, stacked_features

def get_data_loader(dataset, batch_size, vocab=None):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if torch.distributed.is_initialized() else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
                      collate_fn=lambda x: collate_fn(x, max_len=64, vocab=vocab))

def evaluate_model(model, data_loader, vocab, device):
    model.eval()
    predicted_mfe = []
    true_mfe = []
    mse_loss_sum = 0
    num_samples = 0
    noise_std = 0.0
    with torch.no_grad():
        for batch in data_loader:
            src, mfe_targets, stacked_kmer_features, stacked_features = batch
            src, mfe_targets, stacked_kmer_features, stacked_features = src.to(device), mfe_targets.to(device), stacked_kmer_features.to(device), stacked_features.to(device)

            _, _, _, mfe_preds, vq_loss, denoising_loss = model(src, stacked_kmer_features, stacked_features, noise_std=noise_std)
            if isinstance(vq_loss, torch.Tensor) and vq_loss.dim() > 0:
                vq_loss = vq_loss.mean()
            if isinstance(denoising_loss, torch.Tensor) and denoising_loss.dim() > 0:
                denoising_loss = denoising_loss.mean()

            mfe_pred = mfe_preds.squeeze().cpu().numpy()
            predicted_mfe.extend(mfe_pred)
            true_mfe.extend(mfe_targets.cpu().numpy())

            mse_loss = F.mse_loss(mfe_preds.squeeze(), mfe_targets.squeeze()).item()
            mse_loss_sum += mse_loss * mfe_targets.size(0)
            num_samples += mfe_targets.size(0)

    avg_mse_loss = mse_loss_sum / num_samples
    spearman_corr, _ = spearmanr(np.array(true_mfe), np.array(predicted_mfe))
    pearson_corr, _ = pearsonr(np.array(true_mfe), np.array(predicted_mfe))
    r2 = r2_score(true_mfe, predicted_mfe)
    rmse = np.sqrt(mean_squared_error(true_mfe, predicted_mfe))
    mae = mean_absolute_error(true_mfe, predicted_mfe)
    return spearman_corr, pearson_corr, r2, rmse, mae, avg_mse_loss

# ==========================================
# Main entry point and argument parsing
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE with Transformer')
    parser.add_argument('--train_file', type=str, default='./3egfp_pseudo_1/4.7_train_data_GSM3130437_egfp_pseudo_1_BiologyFeatures.csv', help='Input CSV file')
    parser.add_argument('--test_file', type=str, default='./3egfp_pseudo_1/4.7_test_data_GSM3130437_egfp_pseudo_1_BiologyFeatures.csv', help='Input CSV file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=128, help='embed dimension size')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent space dimension size')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of layers in encoder')
    parser.add_argument('--num_decoder_layers', type=int, default=2, help='Number of layers in decoder')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--num_embeddings', type=int, default=32, help='Number of embeddings in the codebook')
    parser.add_argument('--commitment_cost', type=float, default=0.1, help='Commitment cost for VQ-VAE')
    
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to run on (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--output_dir', type=str, default='./Model', help='Base output directory for checkpoints')
    parser.add_argument('--accum_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--max_len', type=int, default=64, help='Max sequence length')
    parser.add_argument('--trial_id', type=int, default=0, help='Trial ID for saving')
    args = parser.parse_args()
    if args.epochs <= 0 or args.epochs % 10 != 0:
        raise ValueError("--epochs must be a positive multiple of 10")

    # ---------------------------------------------------------
    # New logic: dynamically extract the filename prefix and create a dataset-specific output folder
    # ---------------------------------------------------------
    # Extract the filename from the path, for example '4.8_train_data_...'
    train_filename = os.path.basename(args.train_file)
    # Split by '_' and take the first field, for example '4.8'
    file_prefix = train_filename.split('_')[0] 
    
    # Build the base output directory dynamically, for example './Model/4.8'
    base_output_dir = os.path.join(args.output_dir, file_prefix)
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Models and logs will be saved to: {base_output_dir}")
    # ---------------------------------------------------------

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"Target Device has been set to: {device}")

    # Vocabulary and dataset loading
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3, 'A': 4, 'T': 5, 'G': 6, 'C': 7, '<CLS>': 8}
    char_to_idx = vocab 
    vocab_size = len(vocab)
    
    train_dataset = RNADataset(args.train_file, vocab, is_train=False) 
    test_dataset = RNADataset(args.test_file, vocab, is_train=False)
    
    train_loader = get_data_loader(train_dataset, batch_size=args.batch_size, vocab=vocab)
    test_loader = get_data_loader(test_dataset, batch_size=int(args.batch_size/2), vocab=vocab)

    initial_noise_std = 0.01
    max_noise_std = 0.1

    param_grid = {
        'learning_rate': [0.0001],
        'embed_dim': [128, 256],
        'hidden_dim': [128, 256],
        'latent_dim': [128, 256],
    }
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Total number of parameter combinations: {len(combinations)}（manually reduce the grid to speed up the search）")

    best_spearman = -np.inf
    best_params = None
    best_trial_id = -1

    for trial_id, params in enumerate(combinations):
        print(f"\n=== Starting Trial {trial_id} | parameters: {params} ===")
        
        # Replace the hard-coded '4.8' here with the dynamically generated base_output_dir
        trial_dir = os.path.join(base_output_dir, f"trial_{trial_id}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # Add the prefix to the TensorBoard log path to distinguish runs from different datasets
        writer = SummaryWriter(log_dir=f'runs/{file_prefix}_trial_{trial_id}')
        
        model = VAEWithTransformer(
            vocab_size=vocab_size,
            embed_dim=params['embed_dim'],
            hidden_dim=params['hidden_dim'],
            latent_dim=params['latent_dim'],
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            num_heads=args.num_heads,
            num_embeddings=args.num_embeddings,
            commitment_cost=args.commitment_cost,
            kmer_feature_dim=128
        ).to(device)
        
        print(f"Using GPU: {device}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-5, amsgrad=True)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) 
        noise_schedule = cosine_noise_schedule(initial_noise_std, max_noise_std, args.epochs)
        trial_final_metrics = None
        
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                src, mfe_targets, stacked_kmer_features, stacked_features = [x.to(device) for x in batch]
                targets = src
                noise_std = noise_schedule(epoch)
                
                logits, mu, logvar, mfe_preds, vq_loss, denoising_loss = model(src, stacked_kmer_features, stacked_features, noise_std=noise_std)
                
                if isinstance(vq_loss, torch.Tensor) and vq_loss.dim() > 0:
                    vq_loss = vq_loss.mean()
                if isinstance(denoising_loss, torch.Tensor) and denoising_loss.dim() > 0:
                    denoising_loss = denoising_loss.mean()
                
                loss, mfe_loss, _ = loss_function(logits, targets, mu, logvar, vq_loss, denoising_loss,
                                                  mfe_preds, mfe_targets, vocab,
                                                  beta=1.0, mask_weight=0.1, mfe_weight=1.0,
                                                  denoise_weight=1.0, vq_beta=1.0)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            lr_scheduler.step()
            avg_train_loss = total_loss / num_batches
            writer.add_scalar('Train/Loss', avg_train_loss, epoch)
            
            print(
                f"Trial {trial_id} | Epoch {epoch + 1} | "
                f"TrainLoss: {avg_train_loss:.4f}"
            )

            if (epoch + 1) % 10 == 0:
                trial_final_metrics = evaluate_model(model, test_loader, vocab, device)
                spearman_corr, pearson_corr, r2, rmse, mae, avg_mse = trial_final_metrics
                print(
                    f"Test at epoch {epoch + 1} | MSE: {avg_mse:.4f} | "
                    f"Spearman: {spearman_corr:.4f} | Pearson: {pearson_corr:.4f} | "
                    f"R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}"
                )
                writer.add_scalar('Test/Spearman', spearman_corr, epoch)

        spearman_corr, pearson_corr, r2, rmse, mae, avg_mse = trial_final_metrics
        param_str = '_'.join([f"{k}{v}" for k, v in params.items()])
        ckpt_path = os.path.join(
            trial_dir,
            f"final_{param_str}_spear{spearman_corr:.4f}_pear{pearson_corr:.4f}_"
            f"R2{r2:.4f}_RMSE{rmse:.4f}_MAE{mae:.4f}.pth",
        )
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'params': params,
            'spearman': spearman_corr,
            'epoch': args.epochs,
        }, ckpt_path)
        print(f"Final-epoch model saved: {ckpt_path}")

        if spearman_corr > best_spearman:
            best_spearman = spearman_corr
            best_params = params
            best_trial_id = trial_id
            
        writer.close()
        torch.cuda.empty_cache()

    print("\nHyperparameter search completed.")
    print(f"Global best Spearman: {best_spearman:.4f}")
    print(f"Best parameter combination: {best_params}")
    print(f"Best Trial ID: {best_trial_id}")

    # The best-parameter record is also saved in the corresponding dataset-specific folder
    with open(os.path.join(base_output_dir, "best_hyperparameters.txt"), "w") as f:
        f.write(f"Best Spearman: {best_spearman:.4f}\n")
        f.write(f"Best Trial: {best_trial_id}\n")
        f.write(str(best_params))
