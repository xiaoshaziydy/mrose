import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, matthews_corrcoef
from torch.utils.tensorboard import SummaryWriter
import warnings
from Bio import SeqIO
try:
    from ViennaRNA import RNA
except Exception:
    try:
        import RNA
    except Exception:
        RNA = None
warnings.filterwarnings("ignore", category=DeprecationWarning)
import copy
import os
from scipy.stats import spearmanr, pearsonr
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint_sequential
from datetime import datetime, timedelta
import logging
import traceback
import pandas as pd
import random
import optuna
import shutil
import glob
from collections import defaultdict
# Sequence processing functions
def rna_to_idx(rna_seq, codon_table, char_to_idx):
    codons = [rna_seq[i:i+3] for i in range(0, len(rna_seq) - 2, 3)]
    valid_codons = set(codon_table.keys())
    return [char_to_idx.get(codon, char_to_idx['<UNK>']) for codon in codons]
def idx_to_rna(idx_seq):
    return ''.join([idx_to_char.get(idx, '<UNK>') for idx in idx_seq])
def add_special_tokens(rna_seq_list, char_to_idx):
    return [char_to_idx['<SOS>']] + rna_seq_list + [char_to_idx['<EOS>']]
def pad_or_truncate(seq, max_len, char_to_idx):
    if len(seq) > max_len:
        return seq[:max_len]
    else:
        return seq + [char_to_idx['<PAD>']] * (max_len - len(seq))
# Update rna_to_idx and related functions for nucleotides
def nuc_to_idx(nuc_seq, nuc_char_to_idx):
    return [nuc_char_to_idx.get(nuc, nuc_char_to_idx['N']) for nuc in nuc_seq]
def idx_to_nuc(idx_seq):
    return ''.join([nuc_idx_to_char.get(idx, '<UNK>') for idx in idx_seq])
def add_special_tokens_nuc(nuc_seq_list, nuc_char_to_idx):
    return [nuc_char_to_idx['<SOS>']] + nuc_seq_list + [nuc_char_to_idx['<EOS>']]
def pad_or_truncate_nuc(seq, max_len, nuc_char_to_idx):
    if len(seq) > max_len:
        return seq[:max_len]
    else:
        return seq + [nuc_char_to_idx['<PAD>']] * (max_len - len(seq))
def create_mask(seq, mask_prob=0.2):
    mask = np.random.rand(len(seq)) < mask_prob
    mask[0] = mask[-1] = False
    return mask
class RNADataset(Dataset):
    def __init__(self, csv_file, split='train', is_train=True, max_len=2048, codon_table=None, char_to_idx=None, nuc_char_to_idx=None):
        self.nuc_max_len = max_len
        self.max_len = max_len // 3
        self.target_len = self.max_len
        self.rna_sequences = []
        self.nuc_sequences = []
        self.labels = []
        self.is_train = is_train
        self.codon_table = codon_table
        self.char_to_idx = char_to_idx
        self.nuc_char_to_idx = nuc_char_to_idx
        df = pd.read_csv(csv_file)
        df = df[df['Split'] == split].reset_index(drop=True)
       
        for _, row in df.iterrows():
            rna_sequence = row['Sequence'].upper().replace('U', 'T')
            if len(rna_sequence) % 3 != 0:
                rna_sequence = rna_sequence[:-(len(rna_sequence) % 3)]
            if len(rna_sequence) > self.nuc_max_len:
                rna_sequence = rna_sequence[:self.nuc_max_len]
            elif len(rna_sequence) < self.nuc_max_len:
                rna_sequence = rna_sequence + 'N' * (self.nuc_max_len - len(rna_sequence))
            codon_len = len(rna_sequence) // 3
            if codon_len > self.target_len:
                rna_sequence = rna_sequence[:self.target_len * 3]
            elif codon_len < self.target_len:
                rna_sequence = rna_sequence + 'NNN' * (self.target_len - codon_len)
            label = float(row['Value'])
            self.rna_sequences.append(rna_sequence)
            self.nuc_sequences.append(rna_sequence)
            self.labels.append(label)
    def __len__(self):
        return len(self.rna_sequences)
    def __getitem__(self, idx):
        rna_seq = self.rna_sequences[idx]
        nuc_seq = self.nuc_sequences[idx]
        if random.random() < 0.2 and self.is_train:
            codon_idx = random.randint(0, (len(rna_seq) // 3) - 1)
            mutated_codon = random.choice(list(self.codon_table.keys()))
            start_idx = codon_idx * 3
            rna_seq = rna_seq[:start_idx] + mutated_codon + rna_seq[start_idx+3:]
            nuc_seq = rna_seq
       
        rna_seq_list = rna_to_idx(rna_seq, self.codon_table, self.char_to_idx)
        rna_seq_with_special = add_special_tokens(rna_seq_list, self.char_to_idx)
        rna_seq_with_special = pad_or_truncate(rna_seq_with_special, self.max_len, self.char_to_idx)
        input_tensor = torch.tensor(rna_seq_with_special, dtype=torch.long)
       
        nuc_seq_list = nuc_to_idx(nuc_seq, self.nuc_char_to_idx)
        nuc_seq_with_special = add_special_tokens_nuc(nuc_seq_list, self.nuc_char_to_idx)
        nuc_seq_with_special = pad_or_truncate_nuc(nuc_seq_with_special, self.nuc_max_len, self.nuc_char_to_idx)
        nuc_input_tensor = torch.tensor(nuc_seq_with_special, dtype=torch.long)
        return (input_tensor, nuc_input_tensor, self.labels[idx])
# Model components (unchanged classes)
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
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
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
        #out = nn.ReLU(inplace=True)(out)
        out = nn.LeakyReLU(negative_slope=0.01, inplace=True)(out)
        return out
class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
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
        #out = nn.ReLU(inplace=True)(out)
        out = nn.LeakyReLU(negative_slope=0.01, inplace=True)(out)
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
class NoiseAwareBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super(NoiseAwareBatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))
   
    def forward(self, x, noise_std):
        x = self.bn(x)
        noise_factor = self.scale * noise_std + self.shift
        noise_factor = torch.clamp(noise_factor, min=0.1, max=2.0)
        return x * (1 + noise_factor.view(1, -1, 1))
class AttentionGate(nn.Module):
    def __init__(self, in_channels, skip_channels):
        super(AttentionGate, self).__init__()
        self.conv = nn.Conv1d(in_channels + skip_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, skip):
        combined = torch.cat([x, skip], dim=1)
        attention = self.conv(combined)
        attention = self.sigmoid(attention)
        return skip * attention
class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, num_layers, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=0.1)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.conv_in = nn.Conv1d(in_channels, hidden_dim, 1)
        self.conv_out = nn.Conv1d(hidden_dim, out_channels, 1)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        batch_size, c, width = x.size()
        residual = x
        x = self.conv_in(x)
        x = x.permute(2, 0, 1)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        x = self.conv_out(x)
        x = x + residual
        return x
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y
class DoubleConv1DWithTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, num_heads=2, num_layers=1, hidden_dim=96, kernel_size=7):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        padding1 = (kernel_size - 1) // 2
        padding2 = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=padding1) # Removed spectral_norm
        self.bn1 = NoiseAwareBatchNorm1d(hidden_dim)
        self.se1 = SEBlock(hidden_dim)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.transformer = TransformerBlock(hidden_dim, hidden_dim, num_heads, num_layers, hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, out_channels, kernel_size=kernel_size, padding=padding2, dilation=1) # Removed spectral_norm
        self.bn2 = NoiseAwareBatchNorm1d(out_channels)
        self.se2 = SEBlock(out_channels)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.SF = SelfAttention(out_channels)
    def forward(self, x, noise_std):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out, noise_std)
        out = self.se1(out)
        out = self.relu1(out)
        # Use checkpointing for transformer to save memory
        out = checkpoint_sequential([self.transformer], 1, out, use_reentrant=False)
        out = self.conv2(out)
        out = self.bn2(out, noise_std)
        out = self.se2(out)
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
            residual = self.SF(residual)
        out += residual
        out = nn.LeakyReLU(negative_slope=0.01, inplace=True)(out)
        return out
class BottleneckAttention(nn.Module):
    def __init__(self, channels, num_heads=2):
        super(BottleneckAttention, self).__init__()
        self.attention = nn.MultiheadAttention(channels, num_heads, dropout=0.1)
        self.cross_attention = nn.MultiheadAttention(channels, num_heads, dropout=0.1)
        self.norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, channels)
        )
        self.norm2 = nn.LayerNorm(channels)
    def forward(self, x):
        x = x.permute(2, 0, 1)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        cross_output, _ = self.cross_attention(x, x, x)
        x = self.norm(x + cross_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x.permute(1, 2, 0)
class DownWithTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, num_layers=2, hidden_dim=96):
        super().__init__()
        self.maxpool = nn.MaxPool1d(2)
        self.conv = DoubleConv1DWithTransformer(in_channels, out_channels, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
    def forward(self, x, noise_std):
        x = self.maxpool(x)
        x = self.conv(x, noise_std)
        return x
class UpWithTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, num_heads=4, num_layers=2, hidden_dim=96):
        super().__init__()
        self.upsample_mode = 'nearest'
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.attn_gate = AttentionGate(in_channels // 2, skip_channels)
        self.multi_scale = nn.ModuleList([
            nn.Conv1d(skip_channels, in_channels // 2, kernel_size=k, padding=k//2) for k in [3, 5, 7]
        ])
        self.reduce_conv = nn.Conv1d(skip_channels + 2 * in_channels, in_channels // 2, kernel_size=1)
        self.conv = DoubleConv1DWithTransformer(in_channels // 2, out_channels, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
    def forward(self, x1, x2, noise_std):
        x1 = self.up(x1)
        if x1.size(2) != x2.size(2):
            x1 = F.interpolate(x1, size=x2.size(2), mode='nearest')
        x2 = self.attn_gate(x1, x2)
        multi_scale_features = [conv(x2) for conv in self.multi_scale]
        x2_multi = torch.cat(multi_scale_features, dim=1)
        x = torch.cat([x2, x1, x2_multi], dim=1)
        x = self.reduce_conv(x)
        return self.conv(x, noise_std)
class ComplexTransformerUNet1D(nn.Module):
    def __init__(self, in_channels, embed_dim, n_blocks=4, base_channels=64, num_heads=2, num_layers=1, hidden_dim=128, max_len=256):
        super(ComplexTransformerUNet1D, self).__init__()
        self.target_len = max_len // 3
        out_channels = embed_dim
        self.inc = DoubleConv1DWithTransformer(in_channels, base_channels * 8, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        '''
        self.down3 = DoubleConv1DWithTransformer(base_channels * 8, base_channels * 16, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        self.down4 = DoubleConv1DWithTransformer(base_channels * 16, base_channels * 16, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        self.down5 = DoubleConv1DWithTransformer(base_channels * 16, base_channels * 32, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        '''
       
        self.down3 = nn.Sequential(
            nn.AvgPool1d(2),
            DoubleConv1DWithTransformer(base_channels * 8, base_channels * 16, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        )
        self.down4 = nn.Sequential(
            nn.AvgPool1d(2),
            DoubleConv1DWithTransformer(base_channels * 16, base_channels * 16, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        )
       
        self.down5 = nn.Sequential(
            nn.AvgPool1d(2),
            DoubleConv1DWithTransformer(base_channels * 16, base_channels * 32, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        )
        '''
        self.down3 = nn.Sequential(
            nn.AvgPool1d(2),
            DoubleConv1DWithTransformer(base_channels * 8, base_channels * 16, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        )
        self.down4 = nn.Sequential(
            nn.AvgPool1d(2),
            DoubleConv1DWithTransformer(base_channels * 16, base_channels * 16, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        )
       
        #self.down5 = DownWithTransformer(base_channels * 16, base_channels * 32, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        self.down5 = nn.Sequential(
            nn.AvgPool1d(2),
            DoubleConv1DWithTransformer(base_channels * 16, base_channels * 32, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        )
        '''
        self.bottleneck = BottleneckAttention(base_channels * 32, num_heads=2)
        '''
        self.up1 = UpWithTransformer(base_channels * 32, base_channels * 16, skip_channels=base_channels * 16, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        self.up2 = UpWithTransformer(base_channels * 16, base_channels * 8, skip_channels=base_channels * 16, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        self.up3 = UpWithTransformer(base_channels * 8, base_channels * 8, skip_channels=base_channels * 8, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        '''
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            UpWithTransformer(base_channels * 32, base_channels * 16, skip_channels=base_channels * 16, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            UpWithTransformer(base_channels * 16, base_channels * 8, skip_channels=base_channels * 16, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            UpWithTransformer(base_channels * 8, base_channels * 8, skip_channels=base_channels * 8, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        )
        self.outc = nn.Sequential(
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(base_channels * 8, hidden_dim, kernel_size=k, padding=k//2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True)
                ) for k in [1, 3, 5]
            ]),
            nn.Conv1d(hidden_dim * 3, embed_dim, kernel_size=1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            SelfAttention(embed_dim),
            SEBlock(embed_dim),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.ModuleList([
                nn.AdaptiveAvgPool1d(self.target_len),
                nn.AdaptiveMaxPool1d(self.target_len)
            ])
        )
        self.residual_conv = nn.Conv1d(base_channels * 8, embed_dim, kernel_size=1) if base_channels * 8 != embed_dim else None
    def forward(self, x, noise_std):
        x1 = self.inc(x, noise_std)
        x4 = self.down3[0](x1)
        x4 = self.down3[1](x4, noise_std)
        x5 = self.down4[0](x4)
        x5 = self.down4[1](x5, noise_std)
        x6 = self.down5[0](x5)
        x6 = self.down5[1](x6, noise_std)
        '''
        x4 = self.down3(x1, noise_std) # Direct call to DoubleConv1DWithTransformer
        x5 = self.down4(x4, noise_std)
        x6 = self.down5(x5, noise_std)
        '''
        x = self.bottleneck(x6)
       
        x = self.up1[0](x)
        x = self.up1[1](x, x5, noise_std)
        x = self.up2[0](x)
        x = self.up2[1](x, x4,noise_std)
        x = self.up3[0](x)
        x = self.up3[1](x, x1, noise_std)
        '''
        x = self.up1(x, x5, noise_std)
        x = self.up2(x, x4, noise_std)
        x = self.up3(x, x1, noise_std)
        '''
        residual = x1
        multi_scale_outputs = [conv(x) for conv in self.outc[0]]
        x = torch.cat(multi_scale_outputs, dim=1)
        for i, layer in enumerate(self.outc[1:-1]):
            x = layer(x)
        avg_pool = self.outc[-1][0](x)
        max_pool = self.outc[-1][1](x)
        x = (avg_pool + max_pool) / 2.0
        if self.residual_conv is not None:
            if residual.size(2) != x.size(2):
                residual = F.interpolate(residual, size=x.size(2), mode='nearest')
            residual = self.residual_conv(residual)
            x = x + residual
        return x
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
class CodonAwarePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=682):
        super(CodonAwarePositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        codon_pe = torch.zeros(max_len, d_model)
        codon_pos = torch.arange(0, max_len, 3).float().unsqueeze(1)
        codon_div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        for i in range(0, max_len, 3):
            if i < max_len:
                codon_pe[i:i+3, 0::2] = torch.sin(codon_pos[int(i/3)] * codon_div_term)
                codon_pe[i:i+3, 1::2] = torch.cos(codon_pos[int(i/3)] * codon_div_term)
        pe = pe + 0.1 * codon_pe
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"Sequence length {x.size(1)} exceeds maximum length {self.pe.size(1)} in CodonAwarePositionalEncoding")
        x = x + self.pe[:, :x.size(1), :]
        return x
class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(AttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, seq_emb, conv_emb):
        batch_size, seq_len, embed_dim = seq_emb.size()
        combined = torch.stack([seq_emb, conv_emb], dim=0)
        combined = combined.permute(1, 2, 0, 3).reshape(batch_size, seq_len, 2 * embed_dim)
        proj = nn.Linear(2 * embed_dim, embed_dim).to(seq_emb.device)
        combined = proj(combined)
        attn_output, _ = self.attention(combined.transpose(0, 1), combined.transpose(0, 1), combined.transpose(0, 1))
        attn_output = attn_output.transpose(0, 1)
        return self.norm(seq_emb + conv_emb + attn_output)
class CNNFeatureExtractor(nn.Module):
    def __init__(self, embed_dim, num_filters, filter_size):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(embed_dim, num_filters, filter_size, padding=filter_size//2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, filter_size, padding=filter_size//2)
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = F.max_pool1d(x, x.size(2))
        x = x.squeeze(2)
        return x
class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, max_relative_positions=128):
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
        rel_pos_bias = rel_pos_bias.view(seq_len, seq_len, self.num_heads, self.head_dim)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1, 3).sum(dim=-1)
        energy += rel_pos_bias.unsqueeze(0)
        if attn_mask is not None:
            energy += attn_mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            energy = energy.masked_fill(key_padding_mask, float('-inf'))
        attention = torch.softmax(energy / (self.embed_dim ** 0.5), dim=-1)
        attention = self.dropout(attention)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, seq_len, self.embed_dim)
        out = self.fc_out(out)
        out = self.leakyrelu(out)
        return out, attention
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
        relative_positions = torch.arange(seq_length, device=q.device).view(1, seq_length, 1) - torch.arange(seq_length, device=q.device).view(1, 1, seq_length)
        rel_indices = relative_positions.clamp(-self.max_relative_positions + 1, self.max_relative_positions - 1) + self.max_relative_positions
        bucket_indices = self.compute_distance_bucket(rel_indices)
        rel_pos_bias = self.relative_position_bias_table[bucket_indices.view(-1)].view(seq_length, seq_length, -1)
        return rel_pos_bias
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
        #src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        #src2 = self.linear2(self.dropout(F.leaky_relu(self.linear1(src2), negative_slope=0.01)))
        #src = src + src2
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
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True) # Enable batch_first
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
        tgt2 = self.norm2(tgt)
        #tgt2, _ = self.multihead_attn(tgt2, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        #tgt = tgt + tgt2
        #tgt2 = self.norm3(tgt)
        #tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
        #tgt2 = self.linear2(self.dropout(F.leaky_relu(self.linear1(tgt2), negative_slope=0.01)))
        #tgt = tgt + tgt2
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
# Updated VAEWithTransformer with contrastive learning
class VAEWithTransformer(nn.Module):
    def __init__(self, vocab_size, nuc_vocab_size, embed_dim, hidden_dim, latent_dim, num_encoder_layers,
                 num_decoder_layers, num_heads, num_embeddings, commitment_cost, kmer_feature_dim, max_len=256,codon_table = None,
                 char_to_idx=None, nuc_char_to_idx=None):
        super(VAEWithTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.nuc_vocab_size = nuc_vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.codon_table = codon_table
        self.char_to_idx = char_to_idx
        self.nuc_char_to_idx = nuc_char_to_idx
        self.vector_quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.nuc_embedding = nn.Embedding(nuc_vocab_size, embed_dim)
        self.embed_scale = nn.Parameter(torch.tensor(math.sqrt(embed_dim)))
        self.pos_encoder = CodonAwarePositionalEncoding(embed_dim, max_len=self.max_len // 3)
        self.nuc_pos_encoder = PositionalEncoding(embed_dim, self.max_len)
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(embed_dim // 2),
                nn.LeakyReLU(0.01),
                nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(embed_dim)
            ) for k in [3, 5, 7]
        ])
        self.nuc_conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(embed_dim // 2),
                nn.LeakyReLU(0.01),
                nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(embed_dim)
            ) for k in [3, 5, 7]
        ])
        self.conv_fusion = nn.Conv1d(embed_dim * 3, embed_dim, kernel_size=1)
        self.nuc_conv_fusion = nn.Conv1d(embed_dim * 3, embed_dim, kernel_size=1)
        self.attn_fusion = AttentionFusion(embed_dim, num_heads=4)
        self.nuc_attn_fusion = AttentionFusion(embed_dim, num_heads=4)
        encoder_layer = EnhancedEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.2,
            max_relative_positions=8
        )
        self.transformer_encoder = EnhancedTransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.nuc_transformer_encoder = EnhancedTransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.nuc_encoder_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            #nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01)
        )
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, embed_dim)
        self.decoder_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_decoder = CodonAwarePositionalEncoding(embed_dim, max_len=self.max_len // 3)
        decoder_layer = EnhancedTransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            max_relative_positions=8
        )
        self.transformer_decoder = EnhancedTransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.nuc_decoder_embedding = nn.Embedding(nuc_vocab_size, embed_dim)
        self.nuc_pos_decoder = PositionalEncoding(embed_dim, max_len=self.max_len)
        self.nuc_transformer_decoder = EnhancedTransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_nuc_out = nn.Linear(embed_dim, nuc_vocab_size)
        self.fc_base_pred = nn.Linear(embed_dim, vocab_size)
        self.cnn_feature_extractor0 = CNNFeatureExtractor(embed_dim, num_filters=latent_dim, filter_size=3)
        self.cnn_feature_extractor = CNNFeatureExtractor(embed_dim*4, num_filters=latent_dim, filter_size=3)
        self.fc_class = nn.Linear(latent_dim, 1)
        self.unet = ComplexTransformerUNet1D(embed_dim, embed_dim, base_channels=48, hidden_dim=96, max_len=self.max_len)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.nuc_downsample_conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            stride=3,
            padding=0
        )
        nn.init.kaiming_normal_(self.nuc_downsample_conv.weight, mode='fan_in', nonlinearity='relu')
        self.nuc_downsample_conv_initialized = True
    def init_nuc_downsample_conv(self, in_channels):
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=self.embed_dim,
            kernel_size=3,
            stride=3,
            padding=0
        )
        nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')
        return conv
    def downsample_nuc_encoder_output(self, nuc_encoder_output, max_len):
        target_len = max_len // 3
        if nuc_encoder_output.size(1) % 3 != 0:
            padding = (0, 3 - (nuc_encoder_output.size(1) % 3))
            nuc_encoder_output = F.pad(nuc_encoder_output, padding)
        nuc_encoder_output = nuc_encoder_output.permute(0, 2, 1)
        nuc_encoder_output = self.nuc_downsample_conv(nuc_encoder_output)
        nuc_encoder_output = nuc_encoder_output.permute(0, 2, 1)
        if nuc_encoder_output.size(1) > target_len:
            nuc_encoder_output = nuc_encoder_output[:, :target_len, :]
        elif nuc_encoder_output.size(1) < target_len:
            padding = torch.zeros(
                nuc_encoder_output.size(0), target_len - nuc_encoder_output.size(1), nuc_encoder_output.size(2),
                device=nuc_encoder_output.device
            )
            nuc_encoder_output = torch.cat([nuc_encoder_output, padding], dim=1)
        return nuc_encoder_output
    def get_global_representation(self, encoder_output):
        return encoder_output.mean(dim=1)
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    def encode(self, src, max_len=256):
        batch_size, seq_len = src.size()
        target_len = max_len // 3
        if seq_len > target_len:
            src = src[:, :target_len]
        else:
            pad_size = target_len - seq_len
            src = F.pad(src, (0, pad_size), value=self.char_to_idx['<PAD>'])
       
        seq_len = target_len
        src_mask = self._generate_square_subsequent_mask(seq_len).to(src.device)
        src_key_padding_mask = (src == self.char_to_idx['<PAD>'])
        src_emb = self.embedding(src) * self.embed_scale
        src_emb = self.norm1(src_emb)
        src_emb = self.pos_encoder(src_emb)
        src_conv = src_emb.permute(0, 2, 1)
        conv_outputs = [conv(src_conv) for conv in self.conv_layers]
        conv_cat = torch.cat(conv_outputs, dim=1)
        src_conv = self.conv_fusion(conv_cat)
        src_conv = src_conv.permute(0, 2, 1)
        fused_emb = self.attn_fusion(src_emb, src_conv)
        encoder_input = self.encoder_norm(fused_emb)
        encoder_output = self.transformer_encoder(
            encoder_input,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        encoder_output = encoder_input + encoder_output
        encoder_output = self.encoder_norm(encoder_output)
        encoder_output_avg_pooled = encoder_output.mean(dim=1)
        h = self.fc1(encoder_output_avg_pooled)
        mu = self.fc21(h)
        log_var = self.fc22(h)
        return mu, log_var, encoder_output
    def encode_nuc(self, src_nuc, max_len=256):
        batch_size, seq_len = src_nuc.size()
        target_len = max_len
        if seq_len > target_len:
            src_nuc = src_nuc[:, :target_len]
        else:
            pad_size = target_len - seq_len
            src_nuc = F.pad(src_nuc, (0, pad_size), value=self.nuc_char_to_idx['<PAD>'])
       
        seq_len = target_len
        src_mask = self._generate_square_subsequent_mask(seq_len).to(src_nuc.device)
        src_key_padding_mask = (src_nuc == self.nuc_char_to_idx['<PAD>'])
        src_emb = self.nuc_embedding(src_nuc) * self.embed_scale
        src_emb = self.norm1(src_emb)
        src_emb = self.nuc_pos_encoder(src_emb)
        src_conv = src_emb.permute(0, 2, 1)
        conv_outputs = [conv(src_conv) for conv in self.nuc_conv_layers]
        conv_cat = torch.cat(conv_outputs, dim=1)
        src_conv = self.nuc_conv_fusion(conv_cat)
        src_conv = src_conv.permute(0, 2, 1)
        fused_emb = self.nuc_attn_fusion(src_emb, src_conv)
        encoder_input = self.nuc_encoder_norm(fused_emb)
        encoder_output = self.nuc_transformer_encoder(
            encoder_input,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        encoder_output = encoder_input + encoder_output
        encoder_output = self.nuc_encoder_norm(encoder_output)
        return encoder_output
    def decode(self, decoder_input, nuc_decoder_input):
        # Codon decoding
        tgt_seq_len = decoder_input.size(1)
        #h = F.relu(self.fc3(z))
        #h = self.fc4(h).unsqueeze(1).expand(-1, tgt_seq_len, -1)
        emb = decoder_input * math.sqrt(self.decoder_embedding.embedding_dim)
        emb = self.pos_decoder(emb)
        memory = self.transformer_encoder(emb)
        codon_output = self.transformer_decoder(emb, memory)
       
        # Nucleotide decoding
        nuc_tgt_seq_len = nuc_decoder_input.size(1)
        #nuc_h = F.relu(self.fc3(z))
        #nuc_h = self.fc4(nuc_h).unsqueeze(1).expand(-1, nuc_tgt_seq_len, -1)
        nuc_emb = nuc_decoder_input * math.sqrt(self.nuc_decoder_embedding.embedding_dim)
        nuc_emb = self.nuc_pos_decoder(nuc_emb)
        nuc_memory = self.nuc_transformer_encoder(nuc_emb)
        nuc_output = self.nuc_transformer_decoder(nuc_emb, nuc_memory)
       
        return codon_output, nuc_output
    def constrain_logits(self, logits, src, codon_table, char_to_idx, idx_to_char):
        batch_size, seq_len, vocab_size = logits.size()
        constrained_logits = logits.clone()
        aa_to_codons = defaultdict(list)
        for codon, aa in codon_table.items():
            aa_to_codons[aa].append(codon)
        for b in range(batch_size):
            original_codons = [idx_to_char[idx.item()] for idx in src[b] if idx.item() in idx_to_char]
            original_aa = [codon_table.get(codon, '<UNK>') for codon in original_codons if codon not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']]
            for i in range(min(seq_len, len(original_aa))):
                allowed_codons = aa_to_codons[original_aa[i]]
                allowed_indices = [char_to_idx[codon] for codon in allowed_codons if codon in char_to_idx]
                mask = torch.ones(vocab_size, device=logits.device)
                mask[:] = float('-inf')
                for idx in allowed_indices:
                    mask[idx] = 0.0
                constrained_logits[b, i] += mask
        return constrained_logits
    def constrain_nuc_logits(self, nuc_logits, src_nuc, codon_table, nuc_char_to_idx, nuc_idx_to_char, src, char_to_idx, idx_to_char):
        return nuc_logits
    def forward(self, src, src_nuc, noise_std=0.0, max_len=2048):
        mu, log_var, encoder_output = self.encode(src, max_len=max_len)
        nuc_encoder_output = self.encode_nuc(src_nuc, max_len=max_len)
        codon_global_rep = self.get_global_representation(encoder_output)
        nuc_global_rep = self.get_global_representation(nuc_encoder_output)
        nuc_encoder_output = self.downsample_nuc_encoder_output(nuc_encoder_output, max_len)
        #z = self.reparameterize(mu, log_var)
        #z_q, vq_loss, _, _ = self.vector_quantizer(z)
       
        # Codon table (from the original code)
        # Existing noise functions
        def apply_gaussian_noise(tensor, std):
            return tensor + std * torch.randn_like(tensor)
        def apply_uniform_noise(tensor, std):
            return tensor + std * (2 * torch.rand_like(tensor) - 1)
        # New biological mutation noise for codons
        def apply_codon_mutation_noise(tensor, noise_std, codon_table, char_to_idx):
            prob = noise_std / 0.2
            batch_size, seq_len, embed_dim = tensor.size()
            mask = np.random.rand(batch_size, seq_len) < prob
            mask[:, 0] = mask[:, -1] = False # Exclude <SOS> and <EOS>
            mask[:, 1:4] = False # Exclude first codon
            mask[:, -4:-1] = False # Exclude last codon
            mask = torch.tensor(mask, device=tensor.device, dtype=torch.bool)
            aa_to_codons = defaultdict(list)
            for codon, aa in codon_table.items():
                aa_to_codons[aa].append(codon)
            noisy_tensor = tensor.clone()
            for b in range(batch_size):
                for i in range(seq_len):
                    if mask[b, i] and i > 0 and i < seq_len - 1:
                        current_codon = random.choice(list(codon_table.keys()))
                        current_aa = codon_table[current_codon]
                        if torch.rand(1).item() < 0.7:
                            possible_codons = aa_to_codons[current_aa]
                        else:
                            possible_codons = list(codon_table.keys())
                        new_codon = random.choice(possible_codons)
                        new_idx = char_to_idx.get(new_codon, char_to_idx['<UNK>'])
                        noisy_tensor[b, i] = self.embedding(torch.tensor(new_idx, device=tensor.device))
            return noisy_tensor
        # New biological mutation noise for nucleotides
        def apply_nucleotide_mutation_noise(tensor, noise_std, nuc_char_to_idx):
            prob = noise_std / 0.2
            batch_size, seq_len, embed_dim = tensor.size()
            mask = np.random.rand(batch_size, seq_len) < prob
            mask[:, 0] = mask[:, -1] = False
            mask[:, 1:4] = False
            mask[:, -4:-1] = False
            mask = torch.tensor(mask, device=tensor.device, dtype=torch.bool)
            transition_prob = 0.67
            nucleotides = ['A', 'T', 'G', 'C']
            noisy_tensor = tensor.clone()
            for b in range(batch_size):
                for i in range(seq_len):
                    if mask[b, i] and i > 0 and i < seq_len - 1:
                        current_nuc = random.choice(nucleotides)
                        if torch.rand(1).item() < transition_prob:
                            new_nuc = 'G' if current_nuc == 'A' else 'A' if current_nuc == 'G' else 'T' if current_nuc == 'C' else 'C'
                        else:
                            possible_nucs = [n for n in nucleotides if n != current_nuc]
                            new_nuc = random.choice(possible_nucs)
                        new_idx = nuc_char_to_idx.get(new_nuc, nuc_char_to_idx['N'])
                        noisy_tensor[b, i] = self.nuc_embedding(torch.tensor(new_idx, device=tensor.device))
            return noisy_tensor
        # Within the VAEWithTransformer forward method
        # Assuming char_to_idx, idx_to_char, nuc_char_to_idx, nuc_idx_to_char, and codon_table are attributes of the model
        noisy_encoder_output_gauss = apply_gaussian_noise(encoder_output, noise_std)
        denoised_output_gauss = self.unet(noisy_encoder_output_gauss.permute(0, 2, 1), noise_std).permute(0, 2, 1)
        #noisy_encoder_output_uniform = apply_uniform_noise(encoder_output, noise_std)
        #denoised_output_uniform = self.unet(noisy_encoder_output_uniform.permute(0, 2, 1), noise_std).permute(0, 2, 1)
        noisy_encoder_output_bio = apply_codon_mutation_noise(encoder_output, noise_std, self.codon_table, self.char_to_idx)
        denoised_output_bio = self.unet(noisy_encoder_output_bio.permute(0, 2, 1), noise_std).permute(0, 2, 1)
        noisy_nuc_encoder_output_gauss = apply_gaussian_noise(nuc_encoder_output, noise_std)
        denoised_nuc_output_gauss = self.unet(noisy_nuc_encoder_output_gauss.permute(0, 2, 1), noise_std).permute(0, 2, 1)
        #noisy_nuc_encoder_output_uniform = apply_uniform_noise(nuc_encoder_output, noise_std)
        #denoised_nuc_output_uniform = self.unet(noisy_nuc_encoder_output_uniform.permute(0, 2, 1), noise_std).permute(0, 2, 1)
        noisy_nuc_encoder_output_bio = apply_nucleotide_mutation_noise(nuc_encoder_output, noise_std, self.nuc_char_to_idx)
        denoised_nuc_output_bio = self.unet(noisy_nuc_encoder_output_bio.permute(0, 2, 1), noise_std).permute(0, 2, 1)
        # Average the denoised outputs
        decoder_input = (denoised_output_bio + denoised_output_gauss) / 2.0
        nuc_decoder_input = (denoised_nuc_output_bio + denoised_nuc_output_gauss) / 2.0
        #decoder_input_sum = decoder_input + nuc_decoder_input
        #decoder_input_sum = decoder_input_sum.permute(0, 2, 1)
        #decoder_input_features = self.cnn_feature_extractor0(decoder_input_sum)
        codon_output, nuc_output = self.decode(decoder_input, nuc_decoder_input)
        logits = self.fc_out(codon_output)
        nuc_logits = self.fc_nuc_out(nuc_output)
       
        combined_features = torch.cat([
            denoised_output_gauss.permute(0, 2, 1),
            #denoised_output_uniform.permute(0, 2, 1),
            denoised_output_bio.permute(0, 2, 1),
            denoised_nuc_output_gauss.permute(0, 2, 1),
            #denoised_nuc_output_uniform.permute(0, 2, 1),
            denoised_nuc_output_bio.permute(0, 2, 1),
        ], dim=1)
       
        cnn_features = self.cnn_feature_extractor( combined_features)
        class_pred = self.fc_class(cnn_features)
        denoising_loss = (
            F.mse_loss(denoised_output_gauss, encoder_output) +
            #F.mse_loss(denoised_output_uniform, encoder_output) +
            F.mse_loss(denoised_output_bio, encoder_output) +
            F.mse_loss(denoised_nuc_output_gauss, nuc_encoder_output) +
            #F.mse_loss(denoised_nuc_output_uniform, nuc_encoder_output) +
            F.mse_loss(denoised_nuc_output_bio, nuc_encoder_output)
        ) / 4.0
       
        return class_pred, logits, nuc_logits, mu, log_var, denoising_loss, codon_global_rep, nuc_global_rep
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
# Contrastive loss function
def contrastive_loss(codon_rep, nuc_rep, temperature=0.1):
    batch_size = codon_rep.size(0)
    codon_rep = F.normalize(codon_rep, dim=1)
    nuc_rep = F.normalize(nuc_rep, dim=1)
    #print(f"Codon rep norm: {torch.norm(codon_rep, dim=1).mean().item()}")
    #print(f"Nuc rep norm: {torch.norm(nuc_rep, dim=1).mean().item()}")
    similarity_matrix = torch.matmul(codon_rep, nuc_rep.T) / temperature
    diag_elements = torch.diag(similarity_matrix)
    non_diag_elements = similarity_matrix[~torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix.device)]
    #print(f"Diag mean: {diag_elements.mean().item()}, Non-diag mean: {non_diag_elements.mean().item()}")
    labels = torch.arange(batch_size).to(codon_rep.device)
    loss_codon_to_nuc = F.cross_entropy(similarity_matrix, labels)
    loss_nuc_to_codon = F.cross_entropy(similarity_matrix.T, labels)
    loss = (loss_codon_to_nuc + loss_nuc_to_codon) / 2
    #print(f"Contrastive loss: {loss.item()}")
    return loss
# Helper function to translate codon indices to amino acid sequence
def translate_codon_indices(codon_indices, idx_to_char, codon_table):
    """
    Translates a sequence of codon indices to an amino acid sequence.
   
    Args:
        codon_indices: Tensor of shape (batch_size, seq_len) containing codon indices.
        idx_to_char: Dictionary mapping indices to codon strings.
        codon_table: Dictionary mapping codons to amino acids.
   
    Returns:
        List of amino acid sequences (one per batch).
    """
    aa_sequences = []
    for batch_idx in range(codon_indices.size(0)):
        aa_seq = []
        for idx in codon_indices[batch_idx]:
            codon = idx_to_char.get(idx.item(), '<UNK>')
            if codon in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                continue # Skip special tokens
            aa = codon_table.get(codon, '<UNK>')
            aa_seq.append(aa)
        aa_sequences.append(aa_seq)
    return aa_sequences
# New amino acid consistency loss function
def amino_acid_consistency_loss(logits, targets, idx_to_char, codon_table):
    _, predicted_indices = torch.max(logits, dim=-1) # Shape: (batch_size, seq_len)
   
    # Translate predicted and target codon sequences to amino acid sequences
    pred_aa_sequences = translate_codon_indices(predicted_indices, idx_to_char, codon_table)
    target_aa_sequences = translate_codon_indices(targets, idx_to_char, codon_table)
   
    # Compute mismatch penalty using soft probabilities
    batch_size, seq_len, vocab_size = logits.size()
    mismatches = torch.zeros(1, device=logits.device, requires_grad=True)
    total_positions = 0
   
    for b in range(batch_size):
        pred_aa = pred_aa_sequences[b]
        target_aa = target_aa_sequences[b]
        min_len = min(len(pred_aa), len(target_aa))
        if min_len == 0:
            continue
        for i in range(min_len):
            if pred_aa[i] != target_aa[i]:
                # Use log-probabilities to maintain gradient flow
                log_prob = F.log_softmax(logits[b, i], dim=-1)
                target_idx = targets[b, i]
                mismatches = mismatches + (-log_prob[target_idx])
        total_positions += min_len
   
    if total_positions == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
   
    aa_loss = mismatches / total_positions
    return aa_loss
def compute_dispersive_loss(predictions, targets, lambda_weight=1.0):
    """
    Compute the dispersive loss as the variance of predictions across the batch.
    """
    batch_variance = torch.var(predictions, dim=0).mean() # Variance across batch
    return lambda_weight * batch_variance
# Modified loss function to include amino acid consistency loss
def loss_function(logits, nuc_logits, tgt, tgt_nuc, mu, logvar, denoising_loss,
                 value_preds, value_targets, codon_rep, nuc_rep,
                 value_weight=5.0, beta=0.5, denoise_weight=1.0, contrastive_weight=0.5, aa_weight=0.5,
                 codon_table=None, idx_to_char=None):
    # Codon reconstruction loss
    cross_entropy_codon = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
   
    # Downsample tgt_nuc to match nuc_logits sequence length
    batch_size, seq_len = tgt_nuc.size()
    target_len = seq_len // 3
    if seq_len > target_len:
        # Select every third nucleotide or use strided slicing
        # Alternatively, apply the same convolutional downsampling as in downsample_nuc_encoder_output
        # Here, we'll use simple slicing for alignment (assuming codons are groups of 3 nucleotides)
        tgt_nuc_downsampled = tgt_nuc[:, ::3][:,:target_len] # Take every third nucleotide up to target_len
    else:
        # Pad if necessary
        pad_size = target_len - seq_len
        tgt_nuc_downsampled = F.pad(tgt_nuc, (0, pad_size), value=nuc_char_to_idx['<PAD>'])[:, :target_len]
   
    # Nucleotide reconstruction loss
    cross_entropy_nuc = F.cross_entropy(nuc_logits.view(-1, nuc_logits.size(-1)), tgt_nuc_downsampled.view(-1))
   
    # Combine reconstruction losses as L_Diff
    l_diff = (cross_entropy_codon + cross_entropy_nuc) / 2.0
   
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
   
    # Classification loss (cross-entropy)
    value_preds = value_preds.view(-1)
    value_targets = value_targets.view(-1)
    value_loss = F.mse_loss(value_preds, value_targets, reduction='mean')
   
    # Contrastive loss
    contrastive_loss_value = contrastive_loss(codon_rep, nuc_rep, temperature=0.1)
   
    # Amino acid consistency loss
    aa_consistency_loss = amino_acid_consistency_loss(logits, tgt, idx_to_char, codon_table)
   
    # Total loss
    total_loss = (
        l_diff +
        beta * KLD +
        value_weight * value_loss +
        denoise_weight * denoising_loss +
        contrastive_weight * contrastive_loss_value +
        aa_weight * aa_consistency_loss
    )
   
    return total_loss, contrastive_loss_value, aa_consistency_loss, l_diff, value_loss
def cosine_noise_schedule(initial_noise_std, max_noise_std, total_epochs):
    def schedule(epoch):
        t = epoch
        beta_start = initial_noise_std
        beta_end = max_noise_std
        T = total_epochs
        noise_std = beta_start + 0.5 * (beta_end - beta_start) * (1 - np.cos(np.pi * t / T))
        return noise_std
    return schedule
def calculate_accuracy(predicted, targets, mask):
    _, predicted_indices = torch.max(predicted, dim=1)
    correct = (predicted_indices == targets) & mask.bool()
    accuracy = correct.float().sum() / mask.sum().item()
    return accuracy.item()
def collate_fn(batch, max_len=256, codon_vocab=None, nuc_vocab=None):
    (seq_tensors, nuc_seq_tensors, labels) = zip(*batch)
    target_len = max_len // 3
    nuc_target_len = max_len
    max_len_one_hot = target_len
    max_len_nuc_one_hot = nuc_target_len
   
    padded_seqs = torch.full((len(seq_tensors), target_len), codon_vocab['<PAD>'], dtype=torch.long)
    padded_nuc_seqs = torch.full((len(nuc_seq_tensors), nuc_target_len), nuc_vocab['<PAD>'], dtype=torch.long)
   
    for i, (seq, nuc_seq) in enumerate(zip(seq_tensors, nuc_seq_tensors)):
        seq_len = min(seq.size(0), target_len)
        nuc_seq_len = min(nuc_seq.size(0), nuc_target_len)
        padded_seqs[i, :seq_len] = seq[:seq_len]
        padded_nuc_seqs[i, :nuc_seq_len] = nuc_seq[:nuc_seq_len]
   
    return (padded_seqs, padded_nuc_seqs, torch.tensor(labels, dtype=torch.float))
class CollateFnWrapper:
    def __init__(self, max_len, codon_vocab, nuc_vocab):
        self.max_len = max_len
        self.codon_vocab = codon_vocab
        self.nuc_vocab = nuc_vocab
   
    def __call__(self, batch):
        return collate_fn(batch, max_len=self.max_len, codon_vocab=self.codon_vocab, nuc_vocab=self.nuc_vocab)
def get_data_loader(dataset, batch_size, vocab=None, max_len=2048, num_workers=4):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if torch.distributed.is_initialized() else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=CollateFnWrapper(max_len=max_len, codon_vocab=vocab['codon'], nuc_vocab=vocab['nuc']),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
def calculate_sequence_lengths(file_name, vocab):
    lengths = []
    with open(file_name, 'r') as file:
        for line in file:
            if not line.startswith('>'):
                rna_sequence = line.strip().upper()
                codons = [rna_sequence[i:i+3] for i in range(0, len(rna_sequence) - 2, 3)]
                lengths.append(len(codons))
    return lengths
# Modified setup_logging function
def setup_logging(rank, output_dir):
    """
    Set up logging for a specific rank, ensuring unique log files and proper formatting.
   
    Args:
        rank (int): The rank of the process in DDP.
        output_dir (str): Directory to store log files.
   
    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'rank_{rank}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
   
    # Create a new logger instance for this rank
    logger = logging.getLogger(f'rank_{rank}')
    #logger.setLevel(logging.INFO)
   
    # Remove existing handlers to prevent duplicate logs
    logger.handlers = []
   
    # File handler
    file_handler = logging.FileHandler(log_file)
    #file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - Rank %(rank)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
   
    # Stream handler (for console output)
    stream_handler = logging.StreamHandler()
    #stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - Rank %(rank)s - %(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
   
    # Add rank to log messages
    logger = logging.LoggerAdapter(logger, {'rank': rank})
   
    return logger
def cleanup_ddp():
    """Clean up DDP process group and ensure all processes are synchronized."""
    if dist.is_initialized():
        try:
            dist.barrier() # Synchronize before destroying
            dist.destroy_process_group()
            logging.info("DDP process group cleaned up successfully")
        except Exception as e:
            logging.error(f"Error during DDP cleanup: {e}")
            dist.destroy_process_group() # Force cleanup even if barrier fails
def init_ddp(rank, world_size):
    """Initialize DDP with proper settings and increased timeout."""
    try:
        os.environ['NCCL_BLOCKING_WAIT'] = '1'
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        #os.environ['NCCL_DEBUG'] = 'INFO' # Set to TRACE for detailed debugging if needed
        dist.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=3600) # Increased to 1 hour
        )
        torch.cuda.set_device(rank)
        logging.info(f"Rank {rank}: DDP initialized successfully")
    except Exception as e:
        logging.error(f"Rank {rank}: Error initializing DDP: {e}")
        raise
def train_step(model, optimizer, batch, device, noise_std, max_len, codon_table, idx_to_char, accum_steps=2):
    model.train()
    src, src_nuc, labels = batch
    src, src_nuc, labels = src.to(device), src_nuc.to(device), labels.to(device).float()
   
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
   
    micro_batch_size = max(1, src.size(0) // accum_steps)
    for i in range(0, src.size(0), micro_batch_size):
        micro_src = src[i:i+micro_batch_size]
        micro_src_nuc = src_nuc[i:i+micro_batch_size]
        micro_labels = labels[i:i+micro_batch_size]
       
        value_preds, logits, nuc_logits, mu, log_var, denoising_loss, codon_global_rep, nuc_global_rep = model(
            micro_src, micro_src_nuc, noise_std=noise_std, max_len=max_len
        )
        loss, _, _, _, _ = loss_function(
            logits, nuc_logits, micro_src, micro_src_nuc, mu, log_var, denoising_loss,
            value_preds, micro_labels, codon_global_rep, nuc_global_rep,
            value_weight=1.0, beta=1, denoise_weight=1.0, contrastive_weight=1,
            aa_weight=1, codon_table=codon_table, idx_to_char=idx_to_char
        )
        loss = loss / accum_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # Add gradient clipping
        loss_accum += loss.item() * accum_steps
    optimizer.step()
    return loss_accum
def evaluate_model(model, data_loader, device, noise_std, max_len, codon_table, idx_to_char):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
   
    with torch.no_grad():
        for batch in data_loader:
            src, src_nuc, labels = batch
            src, src_nuc, labels = src.to(device), src_nuc.to(device), labels.to(device).float()
           
            value_preds, logits, nuc_logits, mu, log_var, denoising_loss, codon_global_rep, nuc_global_rep = model(
                src, src_nuc, noise_std=noise_std, max_len=max_len
            )
           
            all_preds.append (value_preds.squeeze ().cpu ().numpy ())
            all_labels.append (labels.cpu ().numpy ())
            batch_loss = F.mse_loss (value_preds.squeeze (), labels)
            total_loss += batch_loss.item ()
            num_batches += 1
    all_preds = np.concatenate (all_preds)
    all_labels = np.concatenate (all_labels)
    mse = mean_squared_error (all_labels, all_preds)
    rmse = np.sqrt (mse)
    r2 = r2_score (all_labels, all_preds)
    pearson_corr, _ = pearsonr (all_labels, all_preds)
    spearman_corr, _ = spearmanr (all_labels, all_preds)
    avg_loss = total_loss /num_batches
    return mse, rmse, r2, pearson_corr, spearman_corr, avg_loss

def build_vocabularies():
    """Build codon and nucleotide vocabularies used by the pretrained model."""
    nuc_vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, 'A': 3, 'T': 4, 'G': 5, 'C': 6, 'N': 7}
    nuc_char_to_idx = nuc_vocab
    nuc_idx_to_char = {idx: char for char, idx in nuc_vocab.items()}

    codon_table = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    }
    codon_vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for codon in codon_table.keys():
        codon_vocab[codon] = len(codon_vocab)
    idx_to_char = {idx: char for char, idx in codon_vocab.items()}

    return {
        'nuc_vocab': nuc_vocab,
        'nuc_char_to_idx': nuc_char_to_idx,
        'nuc_idx_to_char': nuc_idx_to_char,
        'nuc_vocab_size': len(nuc_vocab),
        'codon_table': codon_table,
        'codon_vocab': codon_vocab,
        'char_to_idx': codon_vocab,
        'idx_to_char': idx_to_char,
        'vocab_size': len(codon_vocab),
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_state_dict(checkpoint):
    """Support raw state_dict, DDP state_dict, and dict checkpoints."""
    if isinstance(checkpoint, dict):
        for key in ['model_state_dict', 'state_dict', 'model']:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip DDP prefix if needed.
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    return state_dict



def infer_checkpoint_hparams(state_dict):
    """Infer architecture-critical hyperparameters from a checkpoint state_dict.

    This prevents size-mismatch errors when continuing training from a checkpoint
    whose trial parameters differ from the command-line defaults.
    """
    inferred = {}
    if 'embedding.weight' in state_dict:
        inferred['embed_dim'] = int(state_dict['embedding.weight'].shape[1])
    if 'fc1.0.bias' in state_dict:
        inferred['hidden_dim'] = int(state_dict['fc1.0.bias'].shape[0])
    if 'vector_quantizer.weight' in state_dict:
        inferred['num_embeddings'] = int(state_dict['vector_quantizer.weight'].shape[0])
        inferred['latent_dim'] = int(state_dict['vector_quantizer.weight'].shape[1])
    if 'pos_encoder.pe' in state_dict:
        pe = state_dict['pos_encoder.pe']
        # Codon positional encoding is [1, max_len//3, embed_dim].
        if pe.dim() == 3:
            inferred['max_len'] = int(pe.shape[1]) * 3
    if 'nuc_pos_encoder.pe' in state_dict and 'max_len' not in inferred:
        pe = state_dict['nuc_pos_encoder.pe']
        # Nucleotide positional encoding is usually [max_len, 1, embed_dim].
        if pe.dim() == 3:
            inferred['max_len'] = int(pe.shape[0])
    return inferred


def apply_inferred_hparams(args, inferred):
    changed = []
    for key, value in inferred.items():
        old_value = getattr(args, key, None)
        if old_value != value:
            setattr(args, key, value)
            changed.append((key, old_value, value))
    if changed:
        print('Auto-matched model hyperparameters from checkpoint:')
        for key, old_value, value in changed:
            print(f'  {key}: {old_value} -> {value}')
    else:
        print('Command-line model hyperparameters already match the checkpoint.')
    return args

def create_model(args, vocab_info, device):
    model = VAEWithTransformer(
        vocab_size=vocab_info['vocab_size'],
        nuc_vocab_size=vocab_info['nuc_vocab_size'],
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        kmer_feature_dim=128,
        max_len=args.max_len,
        codon_table=vocab_info['codon_table'],
        char_to_idx=vocab_info['char_to_idx'],
        nuc_char_to_idx=vocab_info['nuc_char_to_idx']
    ).to(device)
    return model


def save_training_checkpoint(path, model, optimizer, epoch, args, metrics, best_spearman):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'args': vars(args),
        'metrics': metrics,
        'best_spearman': best_spearman,
    }, path)


def train_single_gpu(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(rank=0, output_dir=args.output_dir)
    set_seed(args.seed)
    if args.epochs <= 0 or args.epochs % 10 != 0:
        raise ValueError("--epochs must be a positive multiple of 10")

    if args.max_len % 3 != 0:
        args.max_len = args.max_len - (args.max_len % 3)
        print(f'Adjusted max_len to {args.max_len}')

    if args.device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')
    print(f'Target Device has been set to: {device}')
    if device.type == 'cuda':
        print(f'Using GPU: {torch.cuda.get_device_name(device)}')

    vocab_info = build_vocabularies()

    data_path = os.path.join(args.data_dir, args.data_name)
    train_dataset = RNADataset(
        data_path,
        split='train',
        is_train=args.train_augmentation,
        max_len=args.max_len,
        codon_table=vocab_info['codon_table'],
        char_to_idx=vocab_info['char_to_idx'],
        nuc_char_to_idx=vocab_info['nuc_char_to_idx']
    )
    test_dataset = RNADataset(
        data_path,
        split='test',
        is_train=False,
        max_len=args.max_len,
        codon_table=vocab_info['codon_table'],
        char_to_idx=vocab_info['char_to_idx'],
        nuc_char_to_idx=vocab_info['nuc_char_to_idx']
    )

    print('Training set = train only')

    train_loader = get_data_loader(
        train_dataset,
        batch_size=args.batch_size,
        vocab={'codon': vocab_info['codon_vocab'], 'nuc': vocab_info['nuc_vocab']},
        max_len=args.max_len,
        num_workers=args.num_workers
    )
    test_loader = get_data_loader(
        test_dataset,
        batch_size=max(1, args.batch_size // 2),
        vocab={'codon': vocab_info['codon_vocab'], 'nuc': vocab_info['nuc_vocab']},
        max_len=args.max_len,
        num_workers=args.num_workers
    )

    model = create_model(args, vocab_info, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print('Training model from random initialization with a fresh optimizer.')

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.scheduler_t0, T_mult=args.scheduler_t_mult, eta_min=args.min_lr
    )
    noise_schedule = cosine_noise_schedule(
        initial_noise_std=args.initial_noise_std,
        max_noise_std=args.max_noise_std,
        total_epochs=max(1, args.epochs)
    )
    writer = SummaryWriter(log_dir=os.path.join(
        args.tensorboard_dir,
        f'single_gpu_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    ))
    test_metrics = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        noise_std = noise_schedule(epoch)

        for batch in train_loader:
            loss = train_step(
                model, optimizer, batch, device, noise_std, args.max_len,
                vocab_info['codon_table'], vocab_info['idx_to_char'], accum_steps=args.accum_steps
            )
            if not np.isfinite(loss):
                print(f'Skip non-finite loss at epoch {epoch + 1}: {loss}')
                continue
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        lr_scheduler.step()
        for param_group in optimizer.param_groups:
            if param_group['lr'] < args.min_lr:
                param_group['lr'] = args.min_lr

        current_epoch = epoch + 1
        writer.add_scalar('Train/Loss', avg_loss, current_epoch)
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], current_epoch)
        print(
            f'Epoch {current_epoch}/{args.epochs} | Train Loss: {avg_loss:.4f} | '
            f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
        )

        if current_epoch % 10 == 0:
            mse, rmse, r2, pearson_corr, spearman_corr, test_loss = evaluate_model(
                model, test_loader, device, 0.0, args.max_len,
                vocab_info['codon_table'], vocab_info['idx_to_char']
            )
            test_metrics = {
                'mse': mse, 'rmse': rmse, 'r2': r2,
                'pearson': pearson_corr, 'spearman': spearman_corr,
                'test_loss': test_loss, 'train_loss': avg_loss,
                'lr': optimizer.param_groups[0]['lr'],
            }
            writer.add_scalar('Test/Spearman', spearman_corr, current_epoch)
            writer.add_scalar('Test/Pearson', pearson_corr, current_epoch)
            writer.add_scalar('Test/R2', r2, current_epoch)
            writer.add_scalar('Test/MSE', mse, current_epoch)
            print(
                f'Test at epoch {current_epoch} | Test Loss: {test_loss:.4f} | '
                f'Spearman: {spearman_corr:.4f} | Pearson: {pearson_corr:.4f} | '
                f'R2: {r2:.4f} | RMSE: {rmse:.4f} | MSE: {mse:.4f}'
            )

    final_path = os.path.join(args.output_dir, 'final_single_gpu.pth')
    save_training_checkpoint(
        final_path, model, optimizer, args.epochs, args,
        test_metrics, test_metrics['spearman']
    )
    print(f'Final-epoch model saved: {final_path}')

    writer.close()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    summary_path = os.path.join(args.output_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('Training initialization: random\n')
        f.write(f'Final epoch: {args.epochs}\n')
        f.write(f'Final Spearman: {test_metrics["spearman"]:.4f}\n')
        f.write(f'Output dir: {args.output_dir}\n')
    print('\nSingle-GPU training finished!')
    print(f'Summary saved to: {summary_path}')



def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def get_plain_data_loader(dataset, batch_size, vocab=None, max_len=2048, num_workers=4, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        collate_fn=CollateFnWrapper(max_len=max_len, codon_vocab=vocab['codon'], nuc_vocab=vocab['nuc']),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )


def save_training_checkpoint_unwrapped(path, model, optimizer, epoch, args, metrics, best_spearman):
    raw_model = model.module if hasattr(model, 'module') else model
    torch.save({
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'args': vars(args),
        'metrics': metrics,
        'best_spearman': best_spearman,
    }, path)


def train_ddp(args):
    """Train a randomly initialized model using DDP on visible GPUs.

    Launch example:
        CUDA_VISIBLE_DEVICES=0,2 torchrun --standalone --nproc_per_node=2 continue_train_ddp_from_trial1_fixed.py ...
    """
    if not torch.cuda.is_available():
        raise RuntimeError('DDP training requires CUDA. torch.cuda.is_available() returned False.')

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed + rank)

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        timeout=timedelta(seconds=3600)
    )
    logger = setup_logging(rank=rank, output_dir=args.output_dir)

    if rank == 0:
        print(f'DDP initialized: world_size={world_size}, visible GPUs={torch.cuda.device_count()}')
        print(f'Rank 0 uses physical CUDA_VISIBLE_DEVICES index mapped to cuda:{local_rank}: {torch.cuda.get_device_name(device)}')
    logger.info(f'Rank {rank}/{world_size} started on local_rank={local_rank}, device={device}')

    try:
        if args.epochs <= 0 or args.epochs % 10 != 0:
            raise ValueError("--epochs must be a positive multiple of 10")

        if args.max_len % 3 != 0:
            args.max_len = args.max_len - (args.max_len % 3)
            if rank == 0:
                print(f'Adjusted max_len to {args.max_len}')

        vocab_info = build_vocabularies()
        data_path = os.path.join(args.data_dir, args.data_name)

        train_dataset = RNADataset(
            data_path,
            split='train',
            is_train=args.train_augmentation,
            max_len=args.max_len,
            codon_table=vocab_info['codon_table'],
            char_to_idx=vocab_info['char_to_idx'],
            nuc_char_to_idx=vocab_info['nuc_char_to_idx']
        )
        test_dataset = RNADataset(
            data_path,
            split='test',
            is_train=False,
            max_len=args.max_len,
            codon_table=vocab_info['codon_table'],
            char_to_idx=vocab_info['char_to_idx'],
            nuc_char_to_idx=vocab_info['nuc_char_to_idx']
        )

        if rank == 0:
            print('Training set = train only')

        train_loader = get_data_loader(
            train_dataset,
            batch_size=args.batch_size,
            vocab={'codon': vocab_info['codon_vocab'], 'nuc': vocab_info['nuc_vocab']},
            max_len=args.max_len,
            num_workers=args.num_workers
        )

        # Full test loader is used only by rank 0. Do not use DistributedSampler for evaluation, otherwise
        # rank 0 would only evaluate a shard of the test set.
        test_loader_rank0 = None
        if rank == 0:
            test_loader_rank0 = get_plain_data_loader(
                test_dataset,
                batch_size=max(1, args.batch_size // 2),
                vocab={'codon': vocab_info['codon_vocab'], 'nuc': vocab_info['nuc_vocab']},
                max_len=args.max_len,
                num_workers=args.num_workers,
                shuffle=False
            )

        model = create_model(args, vocab_info, device)
        if rank == 0:
            print('Training model from random initialization with a fresh optimizer.')
            print(f'Target devices: CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}; each process uses cuda:{local_rank}')

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.scheduler_t0, T_mult=args.scheduler_t_mult, eta_min=args.min_lr
        )
        noise_schedule = cosine_noise_schedule(
            initial_noise_std=args.initial_noise_std,
            max_noise_std=args.max_noise_std,
            total_epochs=max(1, args.epochs)
        )

        writer = None
        if rank == 0:
            writer = SummaryWriter(log_dir=os.path.join(
                args.tensorboard_dir,
                f'ddp_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            ))
        test_metrics = None

        for epoch in range(args.epochs):
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            model.train()
            total_loss = 0.0
            num_batches = 0
            noise_std = noise_schedule(epoch)

            for batch in train_loader:
                loss = train_step(
                    model, optimizer, batch, device, noise_std, args.max_len,
                    vocab_info['codon_table'], vocab_info['idx_to_char'], accum_steps=args.accum_steps
                )
                if not np.isfinite(loss):
                    if rank == 0:
                        print(f'Skip non-finite loss at epoch {epoch + 1}: {loss}')
                    continue
                total_loss += loss
                num_batches += 1

            avg_loss_local = total_loss / max(1, num_batches)
            avg_loss_tensor = torch.tensor([avg_loss_local], dtype=torch.float32, device=device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = float(avg_loss_tensor.item() / world_size)

            lr_scheduler.step()
            for param_group in optimizer.param_groups:
                if param_group['lr'] < args.min_lr:
                    param_group['lr'] = args.min_lr

            current_epoch = epoch + 1

            if rank == 0:
                writer.add_scalar('Train/Loss', avg_loss, current_epoch)
                writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], current_epoch)
                print(
                    f'DDP Epoch {current_epoch}/{args.epochs} | '
                    f'Train Loss: {avg_loss:.4f} | '
                    f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
                )

            if current_epoch % 10 == 0:
                if rank == 0:
                    raw_model = model.module
                    mse, rmse, r2, pearson_corr, spearman_corr, test_loss = evaluate_model(
                        raw_model, test_loader_rank0, device, 0.0, args.max_len,
                        vocab_info['codon_table'], vocab_info['idx_to_char']
                    )
                    test_metrics = {
                        'mse': mse, 'rmse': rmse, 'r2': r2,
                        'pearson': pearson_corr, 'spearman': spearman_corr,
                        'test_loss': test_loss, 'train_loss': avg_loss,
                        'lr': optimizer.param_groups[0]['lr'],
                    }
                    writer.add_scalar('Test/Spearman', spearman_corr, current_epoch)
                    writer.add_scalar('Test/Pearson', pearson_corr, current_epoch)
                    writer.add_scalar('Test/R2', r2, current_epoch)
                    writer.add_scalar('Test/MSE', mse, current_epoch)
                    print(
                        f'Test at epoch {current_epoch} | Test Loss: {test_loss:.4f} | '
                        f'Spearman: {spearman_corr:.4f} | Pearson: {pearson_corr:.4f} | '
                        f'R2: {r2:.4f} | RMSE: {rmse:.4f} | MSE: {mse:.4f}'
                    )
                dist.barrier()

        if rank == 0:
            final_path = os.path.join(args.output_dir, 'final_ddp.pth')
            save_training_checkpoint_unwrapped(
                final_path, model, optimizer, args.epochs, args,
                test_metrics, test_metrics['spearman']
            )
            print(f'Final-epoch model saved: {final_path}')
            if writer is not None:
                writer.close()
            summary_path = os.path.join(args.output_dir, 'training_ddp_summary.txt')
            with open(summary_path, 'w') as f:
                f.write('Training initialization: random\n')
                f.write(f'Final epoch: {args.epochs}\n')
                f.write(f'Final Spearman: {test_metrics["spearman"]:.4f}\n')
                f.write(f'World size: {world_size}\n')
                f.write(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES")}\n')
                f.write(f'Output dir: {args.output_dir}\n')
            print('\nDDP training finished!')
            print(f'Summary saved to: {summary_path}')

    finally:
        ddp_barrier()
        dist.destroy_process_group()
        torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(description='Train the CDS model from random initialization')
    parser.add_argument('--data_name', type=str, default='mRNA_Stability.csv')
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs; must be a positive multiple of 10')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='./output_training')
    parser.add_argument('--tensorboard_dir', type=str, default='./runs')
    parser.add_argument('--accum_steps', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--scheduler_t0', type=int, default=10)
    parser.add_argument('--scheduler_t_mult', type=int, default=2)
    parser.add_argument('--initial_noise_std', type=float, default=0.005)
    parser.add_argument('--max_noise_std', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0', help='Use cuda:0 after setting CUDA_VISIBLE_DEVICES to the physical GPU ID')

    parser.add_argument('--embed_dim', type=int, default=192)
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--latent_dim', type=int, default=384)
    parser.add_argument('--max_len', type=int, default=768)
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_embeddings', type=int, default=32)
    parser.add_argument('--commitment_cost', type=float, default=0.01)

    parser.add_argument('--train_augmentation', action='store_true', help='Enable random codon mutation augmentation during training')
    args = parser.parse_args()
    if args.epochs <= 0 or args.epochs % 10 != 0:
        raise ValueError("--epochs must be a positive multiple of 10")
    return args




# =========================
# Length-matched CDS generation entry point
# =========================

def normalize_cds_sequence(seq):
    """Keep valid DNA/RNA letters, convert U to T, and uppercase."""
    if seq is None:
        return ''
    seq = ''.join(str(seq).split()).upper().replace('U', 'T')
    return ''.join(ch for ch in seq if ch in {'A', 'T', 'G', 'C', 'N'})


def split_codon_part_and_remainder(seq):
    seq = normalize_cds_sequence(seq)
    rem = len(seq) % 3
    if rem == 0:
        return seq, ''
    return seq[:-rem], seq[-rem:]


def translate_to_aa(seq, codon_table):
    seq = normalize_cds_sequence(seq)
    codons = [seq[i:i + 3] for i in range(0, len(seq) - 2, 3)]
    return ''.join(codon_table.get(codon, 'X') for codon in codons)


def process_generation_input_sequence(seq, max_len, vocab_info):
    """Build codon-level and nucleotide-level tensors for the checkpoint-compatible model."""
    seq = normalize_cds_sequence(seq)
    if len(seq) % 3 != 0:
        seq = seq[:len(seq) - (len(seq) % 3)]
    if len(seq) <= 0:
        raise ValueError('Input CDS chunk is empty after codon alignment.')

    # Nucleotide branch: input length is checkpoint max_len, padded with N if needed.
    nuc_seq = seq[:max_len]
    if len(nuc_seq) < max_len:
        nuc_seq = nuc_seq + 'N' * (max_len - len(nuc_seq))
    nuc_idx = nuc_to_idx(nuc_seq, vocab_info['nuc_char_to_idx'])
    nuc_idx = add_special_tokens_nuc(nuc_idx, vocab_info['nuc_char_to_idx'])
    nuc_idx = pad_or_truncate_nuc(nuc_idx, max_len, vocab_info['nuc_char_to_idx'])
    nuc_tensor = torch.tensor([nuc_idx], dtype=torch.long)

    # Codon branch: target length is max_len // 3, padded with NNN/<UNK> if needed.
    codon_seq = seq[:max_len]
    if len(codon_seq) < max_len:
        codon_seq = codon_seq + 'N' * (max_len - len(codon_seq))
    codon_seq = codon_seq[:len(codon_seq) - (len(codon_seq) % 3)]
    codon_idx = rna_to_idx(codon_seq, vocab_info['codon_table'], vocab_info['char_to_idx'])
    codon_idx = add_special_tokens(codon_idx, vocab_info['char_to_idx'])
    codon_idx = pad_or_truncate(codon_idx, max_len // 3, vocab_info['char_to_idx'])
    codon_tensor = torch.tensor([codon_idx], dtype=torch.long)
    return codon_tensor, nuc_tensor


def split_cds_into_chunks(seq, chunk_len):
    seq = normalize_cds_sequence(seq)
    if len(seq) % 3 != 0:
        seq = seq[:len(seq) - (len(seq) % 3)]
    chunk_len = int(chunk_len) - (int(chunk_len) % 3)
    if chunk_len <= 0:
        raise ValueError('chunk_len must be a positive multiple of 3.')
    return [seq[i:i + chunk_len] for i in range(0, len(seq), chunk_len) if seq[i:i + chunk_len]]


def resolve_generation_state_dict(checkpoint):
    state_dict = resolve_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError('Checkpoint did not contain a valid state_dict dictionary.')
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    return state_dict


def sample_length_matched_chunk(model, chunk_seq, device, vocab_info, max_len, num_samples, temperature=1.0):
    """Generate synonymous CDS chunks with exactly len(chunk_seq) nucleotides."""
    chunk_seq = normalize_cds_sequence(chunk_seq)
    if len(chunk_seq) % 3 != 0:
        chunk_seq = chunk_seq[:len(chunk_seq) - (len(chunk_seq) % 3)]
    target_nuc_len = len(chunk_seq)
    target_codon_len = target_nuc_len // 3
    if target_nuc_len <= 0:
        raise ValueError('Chunk length must be positive after codon alignment.')
    if target_nuc_len > max_len:
        raise ValueError(f'Chunk length {target_nuc_len} exceeds max_len {max_len}.')

    input_tensor, nuc_input_tensor = process_generation_input_sequence(chunk_seq, max_len, vocab_info)
    input_tensor = input_tensor.to(device)
    nuc_input_tensor = nuc_input_tensor.to(device)
    original_codons = [chunk_seq[i:i + 3] for i in range(0, target_nuc_len, 3)]

    generated = []
    model.eval()
    with torch.no_grad():
        # Use the checkpoint-compatible forward path so logits are conditioned on the input sequence.
        _, logits, _, _, _, _, _, _ = model(input_tensor, nuc_input_tensor, noise_std=0.0, max_len=max_len)
        constrained_logits = model.constrain_logits(
            logits,
            input_tensor,
            vocab_info['codon_table'],
            vocab_info['char_to_idx'],
            vocab_info['idx_to_char'],
        )[:, :target_codon_len, :]
        probs = F.softmax(constrained_logits / max(float(temperature), 1e-6), dim=-1)

        for _ in range(num_samples):
            sampled = torch.multinomial(
                probs.reshape(-1, probs.size(-1)),
                num_samples=1,
            ).view(1, target_codon_len)
            codons = []
            for pos, idx in enumerate(sampled[0]):
                codon = vocab_info['idx_to_char'].get(int(idx.item()), '<UNK>')
                if codon in {'<PAD>', '<SOS>', '<EOS>', '<UNK>'} or len(codon) != 3:
                    codon = original_codons[pos]
                codons.append(codon)
            seq = ''.join(codons)
            if len(seq) != target_nuc_len:
                seq = seq[:target_nuc_len].ljust(target_nuc_len, 'N')
            generated.append(seq)
    return generated


def generate_length_matched_sequences(model, input_seq, device, vocab_info, max_len, num_samples, temperature=1.0, chunk_len=0, preserve_remainder=True):
    input_seq = normalize_cds_sequence(input_seq)
    codon_part, remainder = split_codon_part_and_remainder(input_seq)
    if len(codon_part) <= 0:
        raise ValueError('input_seq does not contain any complete codons.')

    if remainder:
        print(f'Input length is not divisible by 3. Complete codon part: {len(codon_part)} nt; remainder: {len(remainder)} nt.')
        if preserve_remainder:
            print('The non-codon remainder will be appended unchanged to each generated sequence to preserve exact input length.')
        else:
            print('The non-codon remainder will be dropped.')

    if chunk_len is None or int(chunk_len) <= 0:
        chunk_len = max_len
    chunk_len = min(int(chunk_len), int(max_len))
    chunk_len = chunk_len - (chunk_len % 3)
    if chunk_len <= 0:
        raise ValueError('chunk_len must be positive after codon alignment.')

    if len(codon_part) > max_len:
        print(f'Input CDS length ({len(codon_part)} nt codon part; original {len(input_seq)} nt) exceeds model max_len ({max_len} nt).')
        print(f'Using codon-aligned chunking: chunk_len={chunk_len} nt.')

    chunks = split_cds_into_chunks(codon_part, chunk_len)
    per_sample_segments = [[] for _ in range(num_samples)]
    for chunk_idx, chunk_seq in enumerate(chunks, start=1):
        print(f'Generating chunk {chunk_idx}/{len(chunks)}: {len(chunk_seq)} nt / {len(chunk_seq)//3} codons')
        chunk_generated = sample_length_matched_chunk(
            model=model,
            chunk_seq=chunk_seq,
            device=device,
            vocab_info=vocab_info,
            max_len=max_len,
            num_samples=num_samples,
            temperature=temperature,
        )
        for sample_idx, seq in enumerate(chunk_generated):
            per_sample_segments[sample_idx].append(seq)

    generated = []
    for parts in per_sample_segments:
        seq = ''.join(parts)
        if preserve_remainder and remainder:
            seq += remainder
        generated.append(seq)

    expected_len = len(input_seq) if (preserve_remainder and remainder) else len(codon_part)
    bad = sorted(set(len(seq) for seq in generated if len(seq) != expected_len))
    if bad:
        raise RuntimeError(f'Length matching failed. Expected {expected_len} nt, got lengths {bad}')
    print(f'All generated sequences have matched length: {expected_len} nt')
    return generated, codon_part, remainder


def predict_values_for_sequences(model, sequences, device, vocab_info, max_len, batch_size=16):
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            codon_tensors, nuc_tensors = [], []
            for seq in batch:
                codon_part, _ = split_codon_part_and_remainder(seq)
                if len(codon_part) > max_len:
                    # Prediction model context is fixed; use the first context window for scoring long sequences.
                    codon_part = codon_part[:max_len]
                c, n = process_generation_input_sequence(codon_part, max_len, vocab_info)
                codon_tensors.append(c.squeeze(0))
                nuc_tensors.append(n.squeeze(0))
            input_tensor = torch.stack(codon_tensors).to(device)
            nuc_input_tensor = torch.stack(nuc_tensors).to(device)
            value_preds, _, _, _, _, _, _, _ = model(input_tensor, nuc_input_tensor, noise_std=0.0, max_len=max_len)
            values = value_preds.view(-1).detach().cpu().numpy().tolist()
            preds.extend(values)
    return preds


def write_fasta(path, sequences):
    with open(path, 'w') as f:
        for i, seq in enumerate(sequences, start=1):
            f.write(f'>generated_{i}\n{seq}\n')


def main_generate():
    parser = argparse.ArgumentParser(description='Generate CDS sequences with output length matched to input_seq, including inputs longer than checkpoint max_len.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained checkpoint.')
    parser.add_argument('--input_seq', type=str, default=None, help='Input CDS sequence. Spaces/newlines are ignored.')
    parser.add_argument('--input_fasta', type=str, default=None, help='Optional FASTA file. The first sequence will be used.')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--chunk_len', type=int, default=0, help='Chunk length for long inputs. Default: checkpoint max_len.')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_csv', type=str, default='generated_CDS_length_matched.csv')
    parser.add_argument('--output_fasta', type=str, default='generated_CDS_length_matched.fasta')
    parser.add_argument('--no_preserve_remainder', action='store_true', help='Drop non-codon tail if input length is not divisible by 3.')

    # Defaults are overwritten automatically from checkpoint whenever possible.
    parser.add_argument('--embed_dim', type=int, default=192)
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--latent_dim', type=int, default=384)
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_embeddings', type=int, default=32)
    parser.add_argument('--commitment_cost', type=float, default=0.01)
    parser.add_argument('--max_len', type=int, default=768)
    args = parser.parse_args()

    if args.input_fasta:
        records = list(SeqIO.parse(args.input_fasta, 'fasta'))
        if not records:
            raise ValueError(f'No sequence found in FASTA: {args.input_fasta}')
        input_seq = str(records[0].seq)
    elif args.input_seq:
        input_seq = args.input_seq
    else:
        raise ValueError('Please provide --input_seq or --input_fasta.')

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'Cannot find checkpoint: {args.checkpoint}')

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = resolve_generation_state_dict(checkpoint)
    args = apply_inferred_hparams(args, infer_checkpoint_hparams(state_dict))
    if args.max_len % 3 != 0:
        args.max_len = args.max_len - (args.max_len % 3)
        print(f'Adjusted max_len to {args.max_len}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    vocab_info = build_vocabularies()
    model = create_model(args, vocab_info, device)
    incompatible = model.load_state_dict(state_dict, strict=True)
    print(f'Loaded checkpoint: {args.checkpoint}')
    print(f'Generation model: embed_dim={args.embed_dim}, hidden_dim={args.hidden_dim}, latent_dim={args.latent_dim}, max_len={args.max_len}, device={device}')

    generated, codon_part, remainder = generate_length_matched_sequences(
        model=model,
        input_seq=input_seq,
        device=device,
        vocab_info=vocab_info,
        max_len=args.max_len,
        num_samples=args.num_samples,
        temperature=args.temperature,
        chunk_len=args.chunk_len,
        preserve_remainder=(not args.no_preserve_remainder),
    )

    # Validate amino-acid consistency on the complete-codon part.
    original_aa = translate_to_aa(codon_part, vocab_info['codon_table'])
    pred_values = predict_values_for_sequences(model, generated, device, vocab_info, args.max_len, batch_size=16)
    rows = []
    for i, seq in enumerate(generated, start=1):
        gen_codon_part, gen_remainder = split_codon_part_and_remainder(seq)
        aa = translate_to_aa(gen_codon_part, vocab_info['codon_table'])
        rows.append({
            'rank': i,
            'sequence': seq,
            'length_nt': len(seq),
            'codon_part_length_nt': len(gen_codon_part),
            'remainder': gen_remainder,
            'aa_match': aa == original_aa,
            'pred_value': pred_values[i - 1] if i - 1 < len(pred_values) else np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    write_fasta(args.output_fasta, generated)

    print(f'Original input length: {len(normalize_cds_sequence(input_seq))} nt')
    print(f'Complete codon part length: {len(codon_part)} nt / {len(codon_part)//3} codons')
    if remainder:
        print(f'Non-codon remainder length: {len(remainder)} nt')
    print(f'Saved CSV: {args.output_csv}')
    print(f'Saved FASTA: {args.output_fasta}')
    print('First 5 generated sequences:')
    for row in rows[:5]:
        print(f"Sample {row['rank']}: length={row['length_nt']} nt, aa_match={row['aa_match']}, pred_value={row['pred_value']:.4f}")
        print(row['sequence'])


# Original generator CLI disabled; fused CLI is defined below.




# =====================================================================
# SINGLE-CHECKPOINT GENERATION + SAME-MODEL PREDICTION + CAI/GC TOP10
# Main body is the generation model. The same checkpoint and same model
# instance are used for both candidate generation and pred_value scoring.
# Ranking uses exactly:
# final_score = pred_weight * normalized(pred_value) + cai_weight * CAI
#             + gc_weight * GC_window_score + mfe_weight * MFE_score
#             - internal_stop_penalty
# =====================================================================

codon_usage = {
    "UUU": {"aa": "F", "freq": 17.6, "w": 0.8669950738916257},
    "UUC": {"aa": "F", "freq": 20.3, "w": 1.0},
    "UUA": {"aa": "L", "freq": 7.7, "w": 0.19444444444444445},
    "UUG": {"aa": "L", "freq": 12.9, "w": 0.32575757575757575},
    "UCU": {"aa": "S", "freq": 15.2, "w": 0.7794871794871795},
    "UCC": {"aa": "S", "freq": 17.7, "w": 0.9076923076923077},
    "UCA": {"aa": "S", "freq": 12.2, "w": 0.6256410256410256},
    "UCG": {"aa": "S", "freq": 4.4, "w": 0.22564102564102567},
    "UAU": {"aa": "Y", "freq": 12.2, "w": 0.7973856209150326},
    "UAC": {"aa": "Y", "freq": 15.3, "w": 1.0},
    "UAA": {"aa": "*", "freq": 1.0, "w": 1.0},
    "UAG": {"aa": "*", "freq": 0.8, "w": 1.0},
    "UGU": {"aa": "C", "freq": 10.6, "w": 0.8412698412698413},
    "UGC": {"aa": "C", "freq": 12.6, "w": 1.0},
    "UGA": {"aa": "*", "freq": 1.6, "w": 1.0},
    "UGG": {"aa": "W", "freq": 13.2, "w": 1.0},
    "CUU": {"aa": "L", "freq": 13.2, "w": 0.3333333333333333},
    "CUC": {"aa": "L", "freq": 19.6, "w": 0.494949494949495},
    "CUA": {"aa": "L", "freq": 7.2, "w": 0.18181818181818182},
    "CUG": {"aa": "L", "freq": 39.6, "w": 1.0},
    "CCU": {"aa": "P", "freq": 17.5, "w": 0.8838383838383838},
    "CCC": {"aa": "P", "freq": 19.8, "w": 1.0},
    "CCA": {"aa": "P", "freq": 16.9, "w": 0.8535353535353535},
    "CCG": {"aa": "P", "freq": 6.9, "w": 0.3484848484848485},
    "CAU": {"aa": "H", "freq": 10.9, "w": 0.7218543046357616},
    "CAC": {"aa": "H", "freq": 15.1, "w": 1.0},
    "CAA": {"aa": "Q", "freq": 12.3, "w": 0.35964912280701755},
    "CAG": {"aa": "Q", "freq": 34.2, "w": 1.0},
    "CGU": {"aa": "R", "freq": 4.5, "w": 0.36885245901639346},
    "CGC": {"aa": "R", "freq": 10.4, "w": 0.8524590163934427},
    "CGA": {"aa": "R", "freq": 6.2, "w": 0.5081967213114754},
    "CGG": {"aa": "R", "freq": 11.4, "w": 0.9344262295081968},
    "AUU": {"aa": "I", "freq": 16.0, "w": 0.7692307692307692},
    "AUC": {"aa": "I", "freq": 20.8, "w": 1.0},
    "AUA": {"aa": "I", "freq": 7.5, "w": 0.3605769230769231},
    "AUG": {"aa": "M", "freq": 22.0, "w": 1.0},
    "ACU": {"aa": "T", "freq": 13.1, "w": 0.6931216931216931},
    "ACC": {"aa": "T", "freq": 18.9, "w": 1.0},
    "ACA": {"aa": "T", "freq": 15.1, "w": 0.798941798941799},
    "ACG": {"aa": "T", "freq": 6.1, "w": 0.32275132275132273},
    "AAU": {"aa": "N", "freq": 17.0, "w": 0.8900523560209423},
    "AAC": {"aa": "N", "freq": 19.1, "w": 1.0},
    "AAA": {"aa": "K", "freq": 24.4, "w": 0.7648902821316614},
    "AAG": {"aa": "K", "freq": 31.9, "w": 1.0},
    "AGU": {"aa": "S", "freq": 12.1, "w": 0.6205128205128205},
    "AGC": {"aa": "S", "freq": 19.5, "w": 1.0},
    "AGA": {"aa": "R", "freq": 12.2, "w": 1.0},
    "AGG": {"aa": "R", "freq": 12.0, "w": 0.9836065573770493},
    "GUU": {"aa": "V", "freq": 11.0, "w": 0.39145907473309605},
    "GUC": {"aa": "V", "freq": 14.5, "w": 0.5160142348754448},
    "GUA": {"aa": "V", "freq": 7.1, "w": 0.25266903914590744},
    "GUG": {"aa": "V", "freq": 28.1, "w": 1.0},
    "GCU": {"aa": "A", "freq": 18.4, "w": 0.6642599277978339},
    "GCC": {"aa": "A", "freq": 27.7, "w": 1.0},
    "GCA": {"aa": "A", "freq": 15.8, "w": 0.5703971119133574},
    "GCG": {"aa": "A", "freq": 7.4, "w": 0.26714801444043323},
    "GAU": {"aa": "D", "freq": 21.8, "w": 0.8685258964143426},
    "GAC": {"aa": "D", "freq": 25.1, "w": 1.0},
    "GAA": {"aa": "E", "freq": 29.0, "w": 0.7323232323232323},
    "GAG": {"aa": "E", "freq": 39.6, "w": 1.0},
    "GGU": {"aa": "G", "freq": 10.8, "w": 0.4864864864864865},
    "GGC": {"aa": "G", "freq": 22.2, "w": 1.0},
    "GGA": {"aa": "G", "freq": 16.5, "w": 0.7432432432432433},
    "GGG": {"aa": "G", "freq": 16.5, "w": 0.7432432432432433}
}

def calculate_cai(sequence, codon_usage):
    sequence = sequence.upper().replace('T', 'U')  # Convert to RNA
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3) if len(sequence[i:i+3]) == 3]
    if not codons:
        return 0.0
    log_sum = 0.0
    count = 0
    for codon in codons:
        usage = codon_usage.get(codon, {'w': 0.5})  # Default w=0.5 for unknown
        w = usage['w']
        if w > 0 and usage.get('aa', '') != '*':  # Skip stop codons
            log_sum += math.log(w)
            count += 1
    if count == 0:
        return 0.0
    cai = math.exp(log_sum / count)
    return cai



def compute_cds_metrics_single_checkpoint(seq, vocab_info):
    seq = normalize_cds_sequence(seq)
    codon_part, remainder = split_codon_part_and_remainder(seq)
    codons = [codon_part[i:i + 3] for i in range(0, len(codon_part), 3) if len(codon_part[i:i + 3]) == 3]
    length = max(len(codon_part), 1)
    gc = float((codon_part.count('G') + codon_part.count('C')) / length)
    gc3 = float(sum(1 for c in codons if c[2] in {'G', 'C'}) / len(codons)) if codons else 0.0
    stop_count = int(sum(1 for c in codons if c in {'TAA', 'TAG', 'TGA'}))
    internal_stop_count = int(sum(1 for c in codons[:-1] if c in {'TAA', 'TAG', 'TGA'})) if len(codons) > 1 else 0
    cai = float(calculate_cai(codon_part, codon_usage))
    aa = translate_to_aa(codon_part, vocab_info['codon_table']) if codon_part else ''
    return {
        'sequence': seq,
        'length_nt': len(seq),
        'codon_part_length_nt': len(codon_part),
        'codon_count': len(codons),
        'remainder': remainder,
        'gc': gc,
        'gc3': gc3,
        'CAI': cai,
        'stop_count': stop_count,
        'internal_stop_count': internal_stop_count,
        'start_codon': codons[0] if codons else '',
        'last_codon': codons[-1] if codons else '',
        'aa_sequence': aa,
    }


def minmax_normalize(values):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=float)
    if not finite.any():
        return out
    mn = float(np.nanmin(arr[finite]))
    mx = float(np.nanmax(arr[finite]))
    if abs(mx - mn) < 1e-12:
        out[finite] = 1.0
    else:
        out[finite] = (arr[finite] - mn) / (mx - mn)
    return out


def gc_window_score(gc_values, gc_low=0.40, gc_high=0.65):
    """Score is 1 inside [gc_low, gc_high], then decreases linearly outside."""
    gc = np.asarray(gc_values, dtype=float)
    center = (float(gc_low) + float(gc_high)) / 2.0
    half_width = max((float(gc_high) - float(gc_low)) / 2.0, 1e-6)
    score = 1.0 - np.maximum(0.0, np.abs(gc - center) - half_width) / max(center, 1.0 - center, 1e-6)
    return np.clip(score, 0.0, 1.0)


def calculate_mfe(sequence):
    """Calculate minimum free energy with ViennaRNA. Returns kcal/mol."""
    if RNA is None:
        raise ImportError(
            'ViennaRNA is required for MFE calculation. Install/use an environment that provides `from ViennaRNA import RNA` or `import RNA`.'
        )
    seq = normalize_cds_sequence(sequence).replace('T', 'U')
    if not seq:
        return float('nan')
    _, mfe = RNA.fold(seq)
    return float(mfe)


def mfe_to_score(mfe_values, direction='lower'):
    """Normalize MFE to [0, 1]. direction='lower' means more negative MFE is better."""
    arr = np.asarray(mfe_values, dtype=float)
    if str(direction).lower() == 'lower':
        arr = -arr
    elif str(direction).lower() == 'higher':
        arr = arr
    else:
        raise ValueError("mfe_direction must be 'lower' or 'higher'.")
    return minmax_normalize(arr)


def write_top_fasta(path, top_df):
    with open(path, 'w') as f:
        for _, row in top_df.iterrows():
            mfe_part = f"|MFE={row['MFE']:.6g}|MFE_score={row['MFE_score']:.6g}" if 'MFE' in row and pd.notna(row['MFE']) else ''
            f.write(
                f">rank_{int(row['rank'])}|final_score={row['final_score']:.6g}|"
                f"pred_value={row['pred_value']:.6g}|CAI={row['CAI']:.6g}|gc={row['gc']:.6g}|gc3={row['gc3']:.6g}"
                f"{mfe_part}\n"
            )
            f.write(str(row['sequence']) + '\n')


def load_single_checkpoint_model(args, device):
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = resolve_generation_state_dict(checkpoint)
    args = apply_inferred_hparams(args, infer_checkpoint_hparams(state_dict))
    if args.max_len % 3 != 0:
        args.max_len -= args.max_len % 3
        print(f'Adjusted max_len to {args.max_len}')
    vocab_info = build_vocabularies()
    model = create_model(args, vocab_info, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    print(
        'Loaded single checkpoint model for generation and prediction: '
        f'embed_dim={args.embed_dim}, hidden_dim={args.hidden_dim}, '
        f'latent_dim={args.latent_dim}, max_len={args.max_len}, device={device}'
    )
    return model, vocab_info, args


def predict_values_same_model(model, sequences, device, vocab_info, max_len, batch_size=32):
    """Use the same generation model checkpoint to compute class_pred values.
    Long sequences are scored by length-weighted chunk average.
    """
    chunk_len = int(max_len) - (int(max_len) % 3)
    all_chunks = []
    mapping = []
    for seq_idx, seq in enumerate(sequences):
        codon_part, _ = split_codon_part_and_remainder(seq)
        if not codon_part:
            codon_part = 'NNN'
        chunks = split_cds_into_chunks(codon_part, chunk_len)
        for chunk in chunks:
            all_chunks.append(chunk)
            mapping.append((seq_idx, len(chunk)))

    chunk_preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(all_chunks), batch_size):
            batch = all_chunks[start:start + batch_size]
            codon_tensors, nuc_tensors = [], []
            for seq in batch:
                x_codon, x_nuc = process_generation_input_sequence(seq, max_len, vocab_info)
                codon_tensors.append(x_codon.squeeze(0))
                nuc_tensors.append(x_nuc.squeeze(0))
            x_codon = torch.stack(codon_tensors).to(device)
            x_nuc = torch.stack(nuc_tensors).to(device)
            pred, _, _, _, _, _, _, _ = model(x_codon, x_nuc, noise_std=0.0, max_len=max_len)
            chunk_preds.extend(pred.view(-1).detach().cpu().numpy().astype(float).tolist())

    numer = [0.0 for _ in sequences]
    denom = [0.0 for _ in sequences]
    for (seq_idx, length), value in zip(mapping, chunk_preds):
        numer[seq_idx] += float(value) * float(length)
        denom[seq_idx] += float(length)
    return [numer[i] / denom[i] if denom[i] > 0 else float('nan') for i in range(len(sequences))]


def parse_args_single_checkpoint():
    parser = argparse.ArgumentParser(
        description=(
            'Single-checkpoint CDS generation and selection. The same checkpoint/model is used for '
            'generation and pred_value scoring. Generates 100 candidates by default and selects Top10 with CAI/GC/MFE.'
        )
    )
    parser.add_argument('--checkpoint', type=str, default=os.path.join(os.path.dirname(__file__), 'Model.pth'), help='Same checkpoint used for generation and prediction.')
    parser.add_argument('--input_seq', type=str, default=None, help='Input CDS sequence. Spaces/newlines are ignored.')
    parser.add_argument('--input_fasta', type=str, default=None, help='Optional FASTA file. The first sequence will be used.')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--chunk_len', type=int, default=0, help='Chunk length for long inputs. Default: checkpoint max_len.')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='./generated_single_checkpoint_top10_100')
    parser.add_argument('--all_csv', type=str, default='all_generated_scored.csv')
    parser.add_argument('--top_csv', type=str, default='top10_generated_scored.csv')
    parser.add_argument('--top_fasta', type=str, default='top10_generated_scored.fasta')
    parser.add_argument('--gc_low', type=float, default=0.40)
    parser.add_argument('--gc_high', type=float, default=0.65)
    parser.add_argument('--mfe_direction', type=str, default='lower', choices=['lower', 'higher'],
                        help='MFE ranking direction. lower: more negative MFE is better; higher: less negative MFE is better.')
    parser.add_argument('--pred_weight', type=float, default=0.65)
    parser.add_argument('--cai_weight', type=float, default=0.20)
    parser.add_argument('--gc_weight', type=float, default=0.05)
    parser.add_argument('--mfe_weight', type=float, default=0.10)
    parser.add_argument('--internal_stop_penalty', type=float, default=1.0)
    parser.add_argument('--no_preserve_remainder', action='store_true')
    parser.add_argument('--no_deduplicate', action='store_true')

    # Defaults are automatically overwritten from checkpoint when possible.
    parser.add_argument('--embed_dim', type=int, default=192)
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--latent_dim', type=int, default=384)
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_embeddings', type=int, default=32)
    parser.add_argument('--commitment_cost', type=float, default=0.01)
    parser.add_argument('--max_len', type=int, default=768)
    return parser.parse_args()


def main_single_checkpoint():
    args = parse_args_single_checkpoint()
    if args.input_fasta:
        records = list(SeqIO.parse(args.input_fasta, 'fasta'))
        if not records:
            raise ValueError(f'No sequence found in FASTA: {args.input_fasta}')
        input_seq = str(records[0].seq)
    elif args.input_seq:
        input_seq = args.input_seq
    else:
        raise ValueError('Please provide --input_seq or --input_fasta.')

    input_seq = normalize_cds_sequence(input_seq)
    if not input_seq:
        raise ValueError('Input sequence is empty after normalization.')
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'Cannot find checkpoint: {args.checkpoint}')
    if args.mfe_weight > 0 and RNA is None:
        raise ImportError(
            'MFE scoring is enabled but ViennaRNA cannot be imported. '
            'Please install/use ViennaRNA or set --mfe_weight 0 to disable MFE scoring.'
        )

    os.makedirs(args.output_dir, exist_ok=True)
    all_csv = os.path.join(args.output_dir, args.all_csv)
    top_csv = os.path.join(args.output_dir, args.top_csv)
    top_fasta = os.path.join(args.output_dir, args.top_fasta)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Input length: {len(input_seq)} nt')
    print(f'Using device: {device}')
    print(f'Using one checkpoint for both generation and prediction: {args.checkpoint}')
    print(
        'Scoring formula: final_score = '
        f'{args.pred_weight:.3g} * normalized(pred_value) + '
        f'{args.cai_weight:.3g} * CAI + '
        f'{args.gc_weight:.3g} * GC_window_score + '
        f'{args.mfe_weight:.3g} * MFE_score - internal_stop_penalty'
    )
    print(f'MFE direction: {args.mfe_direction} (lower = more negative MFE is better)')

    model, vocab_info, args = load_single_checkpoint_model(args, device)

    generated, codon_part, remainder = generate_length_matched_sequences(
        model=model,
        input_seq=input_seq,
        device=device,
        vocab_info=vocab_info,
        max_len=args.max_len,
        num_samples=args.num_samples,
        temperature=args.temperature,
        chunk_len=args.chunk_len,
        preserve_remainder=(not args.no_preserve_remainder),
    )

    original_aa = translate_to_aa(codon_part, vocab_info['codon_table'])
    rows = []
    for i, seq in enumerate(generated, start=1):
        seq = normalize_cds_sequence(seq)
        metrics = compute_cds_metrics_single_checkpoint(seq, vocab_info)
        metrics['sample_id'] = i
        metrics['aa_match'] = bool(metrics['aa_sequence'] == original_aa)
        rows.append(metrics)

    df = pd.DataFrame(rows)
    if not args.no_deduplicate:
        before = len(df)
        df = df.drop_duplicates(subset=['sequence']).reset_index(drop=True)
        print(f'Deduplicated candidates: {before} -> {len(df)}')

    print('Predicting generated sequence values with the same checkpoint/model...')
    df['pred_value'] = predict_values_same_model(
        model=model,
        sequences=df['sequence'].tolist(),
        device=device,
        vocab_info=vocab_info,
        max_len=args.max_len,
        batch_size=args.batch_size,
    )

    df['pred_value_norm'] = minmax_normalize(df['pred_value'].to_numpy(dtype=float))
    df['CAI_score'] = np.clip(df['CAI'].to_numpy(dtype=float), 0.0, 1.0)
    df['GC_window_score'] = gc_window_score(df['gc'].to_numpy(dtype=float), args.gc_low, args.gc_high)
    print('Calculating MFE with ViennaRNA...')
    df['MFE'] = [calculate_mfe(seq) for seq in tqdm(df['sequence'].tolist(), desc='MFE')]
    df['MFE_score'] = mfe_to_score(df['MFE'].to_numpy(dtype=float), direction=args.mfe_direction)
    df['internal_stop_penalty_applied'] = (df['internal_stop_count'].to_numpy(dtype=float) > 0).astype(float) * float(args.internal_stop_penalty)
    df['final_score'] = (
        float(args.pred_weight) * df['pred_value_norm'].to_numpy(dtype=float)
        + float(args.cai_weight) * df['CAI_score'].to_numpy(dtype=float)
        + float(args.gc_weight) * df['GC_window_score'].to_numpy(dtype=float)
        + float(args.mfe_weight) * df['MFE_score'].to_numpy(dtype=float)
        - df['internal_stop_penalty_applied'].to_numpy(dtype=float)
    )

    ranked = df.sort_values('final_score', ascending=False).reset_index(drop=True)
    ranked.insert(0, 'global_rank', np.arange(1, len(ranked) + 1))
    top = ranked.head(args.top_k).copy().reset_index(drop=True)
    top.insert(0, 'rank', np.arange(1, len(top) + 1))

    ranked.to_csv(all_csv, index=False)
    top.to_csv(top_csv, index=False)
    write_top_fasta(top_fasta, top)

    print(f'Saved all candidates: {all_csv}')
    print(f'Saved Top{args.top_k} CSV: {top_csv}')
    print(f'Saved Top{args.top_k} FASTA: {top_fasta}')
    print('\nTop candidates:')
    display_cols = ['rank', 'pred_value', 'pred_value_norm', 'CAI', 'gc', 'gc3', 'GC_window_score', 'MFE', 'MFE_score', 'internal_stop_count', 'final_score', 'aa_match', 'length_nt', 'sequence']
    print(top[display_cols].to_string(index=False))


if __name__ == '__main__':
    main_single_checkpoint()
