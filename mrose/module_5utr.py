import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.nn import MultiheadAttention, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math
import copy
import os
import hashlib
import time
import csv
import random
from collections import Counter
from Bio.Seq import Seq

# ==========================================
# Added: cosine noise scheduler
# ==========================================
def cosine_noise_schedule(initial_noise_std, max_noise_std, total_epochs):
    """
    Generate an epoch-dependent noise standard deviation function based on cosine annealing.
    Usage: noise_schedule = cosine_noise_schedule(0.01, 0.1, 30)
          current_noise = noise_schedule(epoch)
    """
    def schedule(epoch):
        t = epoch
        beta_start = initial_noise_std
        beta_end = max_noise_std
        T = total_epochs
        # Cosine annealing/increase schedule
        noise_std = beta_start + 0.5 * (beta_end - beta_start) * (1 - math.cos(math.pi * t / T))
        return noise_std
    return schedule

# ==========================================
# 1. 5' UTR biological feature extraction 
# ==========================================
def compute_features_5utr(utr):
    utr = str(utr).upper()  
    length = len(utr)
    total_codons = length // 3
    if total_codons == 0:
        return {f'codon_{aa}': 0 for aa in ['H','K','I','M','E','F','D','G','T','A','C','Q','S','L','Y','*','P','R','V','W','N']} | \
               {'uAUG': 0, 'uORF': 0, 'CGratio': 0, 'CGperc': 0, 'utrlen_m80': length - 80, 'ATratio': 0}

    try:
        seq = Seq(utr[:total_codons * 3])  
        protein = seq.translate(to_stop=False)
        aas = list(str(protein))
    except Exception:
        aas = []
        total_codons = 0

    aa_count = Counter(aas)
    aa_map = {
        'H': 'codon_H', 'K': 'codon_K', 'I': 'codon_I', 'M': 'codon_M', 'E': 'codon_E',
        'F': 'codon_F', 'D': 'codon_D', 'G': 'codon_G', 'T': 'codon_T', 'A': 'codon_A',
        'C': 'codon_C', 'Q': 'codon_Q', 'S': 'codon_S', 'L': 'codon_L', 'Y': 'codon_Y',
        '*': 'codon_*', 'P': 'codon_P', 'R': 'codon_R', 'V': 'codon_V', 'W': 'codon_W', 'N': 'codon_N'
    }
    aa_freq = {aa_map[aa]: aa_count.get(aa, 0) / total_codons for aa in aa_map}

    uaug = sum(1 for i in range(0, total_codons * 3, 3) if utr[i:i+3] == 'ATG')

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

    nuc_count = Counter(utr)
    c_num, g_num = nuc_count['C'], nuc_count['G']
    a_num, t_num = nuc_count['A'], nuc_count['T']

    cgratio = abs(c_num / g_num - 1) if g_num > 0 else 0
    cgperc = (c_num + g_num) / length if length > 0 else 0
    utrlen_m80 = length - 80
    atratio = abs(a_num / t_num - 1) if t_num > 0 else 0

    result = aa_freq
    result['uAUG'] = uaug
    result['uORF'] = uorf
    result['CGratio'] = cgratio
    result['CGperc'] = cgperc
    result['utrlen_m80'] = utrlen_m80
    result['ATratio'] = atratio
    
    feature_cols = [
        'codon_H', 'codon_K', 'codon_I', 'uAUG', 'CGratio', 'codon_M', 'codon_E',
        'codon_F', 'codon_D', 'codon_G', 'codon_T', 'codon_A', 'codon_C', 'codon_Q',
        'codon_S', 'codon_L', 'codon_Y', 'codon_P', 'codon_R', 'codon_*','codon_V', 'codon_W',
        'codon_N', 'uORF', 'CGperc', 'utrlen_m80', 'ATratio'
    ]
    return [result[col] for col in feature_cols]

# ==========================================
# 2. Core network components
# ==========================================
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
    def __init__(self, in_channels, out_channels, n_blocks=4, base_channels=16, num_heads=2, num_layers=2, hidden_dim=128):
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

class CNNFeatureExtractor(nn.Module):
    def __init__(self, embed_dim, num_filters, filter_size):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(embed_dim, num_filters, filter_size, padding=filter_size//2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, filter_size, padding=filter_size//2)

    def forward(self, x):
        # ---> Fix tensor dimension matching <---
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
            # Original key_padding_mask shape: [Batch, SeqLen]
            # Expand to [Batch, 1, 1, SeqLen] so it can broadcast with energy [Batch, Heads, SeqLen, SeqLen] along the key dimension
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            energy = energy.masked_fill(key_padding_mask.bool(), float('-inf'))

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

# ==========================================
# 3. 5' UTR main model wrapper with dynamic-length updates and fixes
# ==========================================
class Model_5UTR(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, latent_dim=128, num_heads=16, 
                 num_encoder_layers=2, num_decoder_layers=2, num_embeddings=32, commitment_cost=0.1):
        super(Model_5UTR, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        #self.vector_quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_scale = nn.Parameter(torch.tensor(math.sqrt(embed_dim)))
        self.pos_encoder = PositionalEncoding(embed_dim)

        encoder_layer = EnhancedEncoderLayer(d_model=embed_dim, nhead=num_heads, max_relative_positions=8)
        self.transformer_encoder = EnhancedTransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        
        self.cnn_feature_extractor = CNNFeatureExtractor(embed_dim, num_filters=latent_dim, filter_size=3)
        self.unet = ComplexTransformerUNet1D(self.embed_dim, self.embed_dim)
        
        self.feature_embedding = nn.Sequential(
            nn.Linear(27, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(self.embed_dim)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, src):
        batch_size, seq_len = src.size()
        


        # --- Fix bug 1: create src_key_padding_mask from token IDs before embedding ---
        # Boolean mask: [Batch, seq_len], where True indicates PAD
        src_key_padding_mask = (src == 0) 

        # --- Fix bug 2: use the true seq_len instead of len(src) (batch size) ---
        src_mask = self._generate_square_subsequent_mask(seq_len).to(src.device)
        
        # Token embedding and normalization
        src_emb = self.embedding(src) * self.embed_scale
        src_emb = self.norm1(src_emb)
        
        # Transformer encoder call with corrected argument name mask=src_mask
        encoder_output = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        # [Batch, seq_len, embed_dim] -> [Batch, embed_dim]
        encoder_output_avg_pooled = encoder_output.mean(dim=1)
        
        h = F.relu(self.fc1(encoder_output_avg_pooled))
        mu = self.fc21(h)
        log_var = self.fc22(h)
        return mu, log_var, encoder_output

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Add max_len argument interface

    def forward(self, src, stacked_features, noise_std=0.0):
        mu, log_var, encoder_output = self.encode(src)
        #z = self.reparameterize(mu, log_var)
        #z_q, vq_loss, _, _ = self.vector_quantizer(z)
        
        noisy_encoder_output = encoder_output + noise_std * torch.randn_like(encoder_output)
        denoised_output = self.unet(noisy_encoder_output.permute(0, 2, 1))
        
        # ================= NEW: Compute module-specific auxiliary losses =================
        denoising_loss = F.mse_loss(denoised_output.permute(0, 2, 1), encoder_output)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / src.size(0)
        #if isinstance(vq_loss, torch.Tensor): vq_loss = vq_loss.mean()
        
        aux_loss = {
            #'kld': kld_loss,
            #'vq': vq_loss,
            'denoise': denoising_loss
        }
        # ==========================================================
        
        feature_embedded = self.feature_embedding(stacked_features)
        feature_embedded = feature_embedded.unsqueeze(1) 
        
        combined_features = torch.cat([denoised_output.permute(0, 2, 1), feature_embedded], dim=1)
        cnn_features = self.cnn_feature_extractor(denoised_output.permute(0, 2, 1))
        
        # Return features together with the auxiliary-loss dictionary
        return cnn_features, aux_loss