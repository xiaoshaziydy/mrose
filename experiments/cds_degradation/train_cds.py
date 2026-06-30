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
from Bio.Seq import Seq
from collections import Counter, defaultdict
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

# Sequence processing functions
def rna_to_idx(rna_seq, codon_table, char_to_idx):
    codons = [rna_seq[i:i+3] for i in range(0, len(rna_seq) - 2, 3)]
    valid_codons = set(codon_table.keys())
    return [char_to_idx.get(codon, char_to_idx['<UNK>']) for codon in codons]

def idx_to_rna(idx_seq):
    return ''.join([idx_to_char.get(idx, '<UNK>') for idx in idx_seq])

def add_special_tokens(rna_seq_list, char_to_idx):
    return rna_seq_list

def pad_or_truncate(seq, max_len, char_to_idx):
    stop_codons = [char_to_idx['TGA'], char_to_idx['TAA'], char_to_idx['TAG']]
    
    if len(seq) > max_len:
        # Truncate the sequence while reserving the final three bases for a stop codon
        truncated = seq[:max_len-3]
        # Add a stop codon, using TGA as the default here
        return truncated + [char_to_idx['TGA']]
    else:
        # If the sequence is long enough, ensure the final three bases form a stop codon
        if len(seq) >= 3 and seq[-3:] in stop_codons:
            padded = seq + [char_to_idx['<PAD>']] * (max_len - len(seq))
        else:
            # Remove the final three bases if they are not a stop codon, then add a stop codon
            if len(seq) >= 3:
                new_seq = seq[:-3] + [char_to_idx['TGA']]
            else:
                new_seq = seq + [char_to_idx['TGA']]
            padded = new_seq + [char_to_idx['<PAD>']] * (max_len - len(new_seq))
        return padded

# Update rna_to_idx and related functions for nucleotides
def nuc_to_idx(nuc_seq, nuc_char_to_idx):
    return [nuc_char_to_idx.get(nuc, nuc_char_to_idx['N']) for nuc in nuc_seq]

def idx_to_nuc(idx_seq):
    return ''.join([nuc_idx_to_char.get(idx, '<UNK>') for idx in idx_seq])

def add_special_tokens_nuc(nuc_seq_list, nuc_char_to_idx):
    return nuc_seq_list
def pad_or_truncate_nuc(seq, max_len, nuc_char_to_idx):
    stop_codons_nuc = [
        [nuc_char_to_idx['T'], nuc_char_to_idx['G'], nuc_char_to_idx['A']],  # TGA
        [nuc_char_to_idx['T'], nuc_char_to_idx['A'], nuc_char_to_idx['A']],  # TAA  
        [nuc_char_to_idx['T'], nuc_char_to_idx['A'], nuc_char_to_idx['G']]   # TAG
    ]
    
    if len(seq) > max_len:
        # Truncate the sequence while reserving the final three positions for a stop codon
        if max_len < 3:
            return seq[:max_len]
        truncated = seq[:max_len-3]
        return truncated + stop_codons_nuc[0]  # Use TGA as the default stop codon
    else:
        # Ensure the sequence ends with a stop codon
        if len(seq) >= 3 and seq[-3:] in stop_codons_nuc:
            padded = seq + [nuc_char_to_idx['<PAD>']] * (max_len - len(seq))
        else:
            # Remove the final three nucleotides if they are not a stop codon, then add a stop codon
            if len(seq) >= 3:
                new_seq = seq[:-3] + stop_codons_nuc[0]
            else:
                new_seq = seq + stop_codons_nuc[0]
            # If the new sequence length exceeds max_len, truncate it
            if len(new_seq) > max_len:
                new_seq = new_seq[:max_len]
            padded = new_seq + [nuc_char_to_idx['<PAD>']] * (max_len - len(new_seq))
        return padded

def create_mask(seq, mask_prob=0.2):
    mask = np.random.rand(len(seq)) < mask_prob
    mask[0] = mask[-1] = False
    return mask

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
    # Do not count terminal segments without a stop symbol
    
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

class RNADataset(Dataset):
    def __init__(self, csv_file, is_train=True, max_len=256, codon_table=None, char_to_idx=None, nuc_char_to_idx=None):
        self.nuc_max_len = max_len  # Nucleotide max length
        self.max_len = max_len // 3  # Codon max length
        self.target_len = self.max_len  # Codon target length
        self.rna_sequences = []
        self.nuc_sequences = []
        self.features = []
        self.labels = []  # degradation_rate
        self.is_train = is_train
        self.codon_table = codon_table
        self.char_to_idx = char_to_idx
        self.nuc_char_to_idx = nuc_char_to_idx
        
        # Read CSV file
        df = pd.read_csv(csv_file)
        sequences = df['Sequence'].tolist()
        labels = df['Value'].astype(float).tolist()  # Use degradation_rate as label

        for rna_sequence, label in zip(sequences, labels):
            rna_sequence = rna_sequence.upper().replace('U', 'T')
            # Ensure nucleotide length is divisible by 3
            original_rna = rna_sequence
            if len(rna_sequence) % 3 != 0:
                rna_sequence = rna_sequence[:-(len(rna_sequence) % 3)]
            # Compute features on original unpadded sequence
            feat_dict = compute_features(original_rna)
            feat_list = [feat_dict.get(f'codon_{aa}', 0) for aa in ['H','K','I','M','E','F','D','G','T','A','C','Q','S','L','Y','*','P','R','V','W','N']] + \
                        [feat_dict['uAUG'], feat_dict['uORF'], feat_dict['CGratio'], feat_dict['CGperc'], feat_dict['utrlen_m80'], feat_dict['ATratio']]
            self.features.append(feat_list)
            if len(rna_sequence) > self.nuc_max_len:
                rna_sequence = rna_sequence[:self.nuc_max_len]
            elif len(rna_sequence) < self.nuc_max_len:
                rna_sequence = rna_sequence + 'N' * (self.nuc_max_len - len(rna_sequence))
            codon_len = len(rna_sequence) // 3
            if codon_len > self.target_len:
                rna_sequence = rna_sequence[:self.target_len * 3]
            elif codon_len < self.target_len:
                rna_sequence = rna_sequence + 'NNN' * (self.target_len - codon_len)
            
            self.rna_sequences.append(rna_sequence)  # Codon input
            self.nuc_sequences.append(rna_sequence)  # Raw nucleotide input
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

        feat_tensor = torch.tensor(self.features[idx], dtype=torch.float)

        return (input_tensor, nuc_input_tensor, self.labels[idx], feat_tensor)

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
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=padding1)  # Removed spectral_norm
        self.bn1 = NoiseAwareBatchNorm1d(hidden_dim)
        self.se1 = SEBlock(hidden_dim)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.transformer = TransformerBlock(hidden_dim, hidden_dim, num_heads, num_layers, hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, out_channels, kernel_size=kernel_size, padding=padding2, dilation=1)  # Removed spectral_norm
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
        x4 = self.down3(x1, noise_std)  # Direct call to DoubleConv1DWithTransformer
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
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)  # Enable batch_first
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
        self.num_features = 27
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
        self.fc_class = nn.Linear(latent_dim, 1)  # Regression head for degradation_rate
        self.unet = ComplexTransformerUNet1D(embed_dim, embed_dim, base_channels=64, hidden_dim=128, max_len=self.max_len)
        self.unet1 = ComplexTransformerUNet1D(embed_dim, embed_dim, base_channels=64, hidden_dim=128, max_len=self.max_len*3)

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
        conv_outputs = [o.permute(0, 2, 1) for o in conv_outputs]
        src_conv = sum(conv_outputs)
        fused_emb = src_emb + src_conv
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
        conv_outputs = [o.permute(0, 2, 1) for o in conv_outputs]
        src_conv = sum(conv_outputs)
        fused_emb = src_emb + src_conv
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

    def forward(self, src, src_nuc, features, noise_std=0.0, max_len=2048):
        mu, log_var, encoder_output = self.encode(src, max_len=max_len)
        nuc_encoder_output = self.encode_nuc(src_nuc, max_len=max_len)
        codon_global_rep = self.get_global_representation(encoder_output)
        nuc_global_rep = self.get_global_representation(nuc_encoder_output)

        #nuc_encoder_output = self.downsample_nuc_encoder_output(nuc_encoder_output, max_len)

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
            #mask[:, 0] = mask[:, -1] = False  # Exclude <SOS> and <EOS>
            #mask[:, 1:4] = False  # Exclude first codon
            #mask[:, -4:-1] = False  # Exclude last codon
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
            #mask[:, 0] = mask[:, -1] = False
            #mask[:, 1:4] = False
            #mask[:, -4:-1] = False
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
        denoised_nuc_output_gauss = self.unet1(noisy_nuc_encoder_output_gauss.permute(0, 2, 1), noise_std).permute(0, 2, 1)
        #noisy_nuc_encoder_output_uniform = apply_uniform_noise(nuc_encoder_output, noise_std)
        #denoised_nuc_output_uniform = self.unet(noisy_nuc_encoder_output_uniform.permute(0, 2, 1), noise_std).permute(0, 2, 1)
        noisy_nuc_encoder_output_bio = apply_nucleotide_mutation_noise(nuc_encoder_output, noise_std, self.nuc_char_to_idx)
        denoised_nuc_output_bio = self.unet1(noisy_nuc_encoder_output_bio.permute(0, 2, 1), noise_std).permute(0, 2, 1)

        # Average the denoised outputs
        decoder_input = (denoised_output_bio + denoised_output_gauss) / 2.0
        nuc_decoder_input = (denoised_nuc_output_bio + denoised_nuc_output_gauss) / 2.0
        #decoder_input_sum = decoder_input + nuc_decoder_input
        #decoder_input_sum = decoder_input_sum.permute(0, 2, 1)
        #decoder_input_features = self.cnn_feature_extractor0(decoder_input_sum)

        codon_output, nuc_output = self.decode(decoder_input, nuc_decoder_input)
        logits = self.fc_out(codon_output)
        nuc_logits = self.fc_nuc_out(nuc_output)
        
        denoised_nuc_output_gauss_down = self.downsample_nuc_encoder_output(denoised_nuc_output_gauss, max_len)
        denoised_nuc_output_bio_down = self.downsample_nuc_encoder_output(denoised_nuc_output_bio, max_len)


        combined_features = torch.cat([
            denoised_output_gauss.permute(0, 2, 1),
            denoised_output_bio.permute(0, 2, 1),
            denoised_nuc_output_gauss_down.permute(0, 2, 1),
            denoised_nuc_output_bio_down.permute(0, 2, 1),
        ], dim=1)
        
        cnn_features = self.cnn_feature_extractor(combined_features)
        #enhanced_features = torch.cat([cnn_features, features], dim=1)
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
                continue  # Skip special tokens
            aa = codon_table.get(codon, '<UNK>')
            aa_seq.append(aa)
        aa_sequences.append(aa_seq)
    return aa_sequences

# New amino acid consistency loss function
def amino_acid_consistency_loss(logits, targets, idx_to_char, codon_table):
    _, predicted_indices = torch.max(logits, dim=-1)  # Shape: (batch_size, seq_len)
    
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
    batch_variance = torch.var(predictions, dim=0).mean()  # Variance across batch
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
        tgt_nuc_downsampled = tgt_nuc[:, ::3][:,:target_len]  # Take every third nucleotide up to target_len
    else:
        # Pad if necessary
        pad_size = target_len - seq_len
        tgt_nuc_downsampled = F.pad(tgt_nuc, (0, pad_size), value=nuc_char_to_idx['<PAD>'])[:, :target_len]
    
    # Nucleotide reconstruction loss
    cross_entropy_nuc = F.cross_entropy(nuc_logits.view(-1, nuc_logits.size(-1)), tgt_nuc.view(-1))
    
    # Combine reconstruction losses as L_Diff
    l_diff = (cross_entropy_codon + cross_entropy_nuc) / 2.0
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Regression loss (MSE for degradation_rate)
    value_loss = F.mse_loss(value_preds.view(-1), value_targets.view(-1))

    
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
        contrastive_weight * contrastive_loss_value 
        +aa_weight * aa_consistency_loss
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
    (seq_tensors, nuc_seq_tensors, labels, feat_tensors) = zip(*batch)
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
    
    return (padded_seqs, padded_nuc_seqs, torch.tensor(labels, dtype=torch.float), torch.stack(feat_tensors))

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
            dist.barrier()  # Synchronize before destroying
            dist.destroy_process_group()
            logging.info("DDP process group cleaned up successfully")
        except Exception as e:
            logging.error(f"Error during DDP cleanup: {e}")
            dist.destroy_process_group()  # Force cleanup even if barrier fails

def init_ddp(rank, world_size):
    """Initialize DDP with proper settings and increased timeout."""
    try:
        os.environ['NCCL_BLOCKING_WAIT'] = '1'
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        #os.environ['NCCL_DEBUG'] = 'INFO'  # Set to TRACE for detailed debugging if needed
        dist.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=3600)  # Increased to 1 hour
        )
        torch.cuda.set_device(rank)
        logging.info(f"Rank {rank}: DDP initialized successfully")
    except Exception as e:
        logging.error(f"Rank {rank}: Error initializing DDP: {e}")
        raise

def train_step(model, optimizer, batch, device, noise_std, max_len, codon_table, idx_to_char, accum_steps=2):
    model.train()
    src, src_nuc, labels, features = batch
    src, src_nuc, labels, features = src.to(device), src_nuc.to(device), labels.to(device).float(), features.to(device)
    
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    
    micro_batch_size = max(1, src.size(0) // accum_steps)
    for i in range(0, src.size(0), micro_batch_size):
        micro_src = src[i:i+micro_batch_size]
        micro_src_nuc = src_nuc[i:i+micro_batch_size]
        micro_labels = labels[i:i+micro_batch_size]
        micro_features = features[i:i+micro_batch_size]
        
        value_preds, logits, nuc_logits, mu, log_var, denoising_loss, codon_global_rep, nuc_global_rep = model(
            micro_src, micro_src_nuc, micro_features, noise_std=noise_std, max_len=max_len
        )
        loss, _, _, _, _ = loss_function(
            logits, nuc_logits, micro_src, micro_src_nuc, mu, log_var, denoising_loss,
            value_preds, micro_labels, codon_global_rep, nuc_global_rep,
            value_weight=1.0, beta=1, denoise_weight=1.0, contrastive_weight=1,
            aa_weight=1, codon_table=codon_table, idx_to_char=idx_to_char
        )
        loss = loss / accum_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Add gradient clipping

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
            src, src_nuc, labels, features = batch
            src, src_nuc, labels, features = src.to(device), src_nuc.to(device), labels.to(device).float(), features.to(device)
            
            value_preds, logits, nuc_logits, mu, log_var, denoising_loss, codon_global_rep, nuc_global_rep = model(
                src, src_nuc, features, noise_std=noise_std, max_len=max_len
            )
            
            all_preds.append(np.atleast_1d(value_preds.squeeze().cpu().numpy()))
            all_labels.append(np.atleast_1d(labels.cpu().numpy()))
            batch_loss = F.mse_loss(value_preds.view(-1), labels.view(-1))
            total_loss += batch_loss.item()
            num_batches += 1
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    spearman_corr, _ = spearmanr(all_preds, all_labels)
    pearson_corr, _ = pearsonr(all_preds, all_labels)
    r2 = r2_score(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    mae = np.mean(np.abs(all_preds - all_labels))
    avg_loss = total_loss / num_batches
    
    return spearman_corr, pearson_corr, r2, rmse, mae, avg_loss

def train_and_evaluate(args,train_csv,test_csv, rank, world_size, device, param_combo, trial_id, logger):
    
    logger.info(f"Trial {trial_id}, Rank {rank}: {param_combo}")

    try:
        # Adjust max_len
        if args.max_len % 3 != 0:
            args.max_len = args.max_len - (args.max_len % 3)
            logger.info(f"Rank {rank}: Adjusted max_len to {args.max_len}")

        # Define vocabularies
        nuc_vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, 'A': 3, 'T': 4, 'G': 5, 'C': 6, 'N': 7}
        nuc_char_to_idx = nuc_vocab
        nuc_idx_to_char = {idx: char for char, idx in nuc_vocab.items()}
        nuc_vocab_size = len(nuc_vocab)

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
        codon_index = len(codon_vocab)
        for codon in codon_table.keys():
            codon_vocab[codon] = codon_index
            codon_index += 1
        char_to_idx = codon_vocab
        idx_to_char = {idx: char for char, idx in codon_vocab.items()}
        vocab_size = len(codon_vocab)

        # Load datasets
        train_dataset = RNADataset(
            train_csv,
            is_train=False,
            max_len=args.max_len,
            codon_table=codon_table,
            char_to_idx=char_to_idx,
            nuc_char_to_idx=nuc_char_to_idx
        )

        test_dataset = RNADataset(
            test_csv,
            is_train=False,
            max_len=args.max_len,
            codon_table=codon_table,
            char_to_idx=char_to_idx,
            nuc_char_to_idx=nuc_char_to_idx
        )

        # Create data loaders
        train_loader = get_data_loader(
            train_dataset,
            batch_size=args.batch_size,
            vocab={'codon': codon_vocab, 'nuc': nuc_vocab},
            max_len=args.max_len,
            num_workers=args.num_workers
        )
        test_loader = get_data_loader(
            test_dataset,
            batch_size=args.batch_size,
            vocab={'codon': codon_vocab, 'nuc': nuc_vocab},
            max_len=args.max_len,
            num_workers=args.num_workers
        )

        # Initialize model
        model = VAEWithTransformer(
            vocab_size=vocab_size,
            nuc_vocab_size=nuc_vocab_size,
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
            codon_table=codon_table,
            char_to_idx=char_to_idx,
            nuc_char_to_idx=nuc_char_to_idx
        ).to(device)

        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        noise_schedule = cosine_noise_schedule(initial_noise_std=0.01, max_noise_std=0.1, total_epochs=args.epochs)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=param_combo['milestones'], gamma=0.1)

        # Training loop
        test_metrics = None
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0
            #for batch in train_loader:
            for batch in tqdm(train_loader, desc=f'Trial {trial_id} Epoch {epoch+1}/{args.epochs}'):  
    
                logger.info(f"Rank {rank}: Processing batch in epoch {epoch+1}, trial {trial_id}")
                noise_std = noise_schedule(epoch)
                loss = train_step(
                    model, optimizer, batch, device, noise_std, args.max_len, codon_table, idx_to_char,
                    accum_steps=args.accum_steps
                )
                total_loss += loss
                num_batches += 1
            avg_loss = total_loss / num_batches
            
            lr_scheduler.step()
            for param_group in optimizer.param_groups:  
                if param_group['lr'] < 0.000005:  
                    param_group['lr'] = 0.000005

            logger.info(
                f'Epoch {epoch + 1}/{args.epochs} | Rank {rank} | '
                f'Train Loss: {avg_loss:.4f}'
            )

            if (epoch + 1) % 10 == 0:
                test_metrics = evaluate_model(
                    model, test_loader, device, 0, args.max_len,
                    codon_table, idx_to_char
                )
                spearman_corr, pearson_corr, r2, rmse, mae, test_loss = test_metrics
                print(
                    f'Test at epoch {epoch + 1}/{args.epochs} | Rank {rank} | '
                    f'Spearman: {spearman_corr:.4f} | Pearson: {pearson_corr:.4f} | '
                    f'R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}'
                )

        spearman_corr, pearson_corr, r2, rmse, mae, test_loss = test_metrics
        if rank == 0:
            checkpoint_path = os.path.join(
                args.output_dir,
                f'trial_{trial_id}_final_epoch_{args.epochs}_spearman_{spearman_corr:.4f}.pth'
            )
            torch.save(model.module.state_dict(), checkpoint_path)
            logger.info(f'Trial {trial_id}: Saved final checkpoint: {checkpoint_path}')

        return spearman_corr, pearson_corr, test_loss


    except Exception as e:
        logger.error(f"Trial {trial_id + 1}, Rank {rank}: Failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
def hyperparameter_search():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='promoter_tata', help='Input CSV file containing all splits')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--accum_steps', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_trials', type=int, default=3)
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_embeddings', type=int, default=32)
    parser.add_argument('--commitment_cost', type=float, default=0.01)
    args = parser.parse_args()
    if args.epochs <= 0 or args.epochs % 10 != 0:
        raise ValueError("--epochs must be a positive multiple of 10")

    # Define hyperparameter search space
    param_grid = {
        'learning_rate': [0.0005],
        'embed_dim': [192],
        'hidden_dim': [192,384],
        'latent_dim': [384],
        'max_len': [768],
        'milestones': [
            [5, 7, 9],  # current setting
            #[6, 12, 18],  # uniform distribution
            #[8, 16, 22],  # back-loaded distribution
            #[3, 8, 15],   # front-loaded distribution
            #[2, 5, 20]   # middle-position adjustment
        ]
    }

    '''
    param_grid = {
        'learning_rate': [0.0005],  # 0.001 caused instability in Trial 3
        'embed_dim': [256],            # 64 underperformed in Trial 3
        'hidden_dim': [256],      # All performed well, keep all
        'latent_dim': [128],       # All performed well, keep all
        'max_len': [64]                # 512 led to instability, 256 untested
    }
    '''
    # Get rank and world size
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    logger = setup_logging(rank, args.output_dir)

    # Initialize DDP once at the start
    init_ddp(rank, world_size)

    try:
        # Set random seeds for reproducibility
        '''
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        random.seed(42)
        '''
        # Generate trials on rank 0 and broadcast
        if rank == 0:
            keys = param_grid.keys()
            trials = []
            for _ in range(args.num_trials):
                trial_params = {key: random.choice(param_grid[key]) for key in keys}
                trials.append(trial_params)
        else:
            trials = None
        trials_list = [trials]
        dist.barrier()  # Synchronize before broadcasting
        dist.broadcast_object_list(trials_list, src=0)
        trials = trials_list[0]

        # Log results
        results = []

        # Load datasets
        train_csv = "./Data/train.csv"
        test_csv = "./Data/test.csv"
        for trial_id, param_combo in enumerate(trials):
            print(f"Rank {rank}: Starting trial {trial_id + 1}/{args.num_trials}: {param_combo}")
            torch.cuda.empty_cache()
            try:
                # Update args with hyperparameter combination
                args.learning_rate = param_combo['learning_rate']
                args.embed_dim = param_combo['embed_dim']
                args.hidden_dim = param_combo['hidden_dim']
                args.latent_dim = param_combo['latent_dim']
                args.max_len = param_combo['max_len']

                device = torch.device(f'cuda:{rank}')
                print(args)
                # Run training and evaluation

                spearman, pearson, test_loss = train_and_evaluate(args, train_csv,test_csv, rank, world_size, device, param_combo, trial_id + 1, logger)

            except Exception as e:
                logger.error(f"Trial {trial_id + 1}, Rank {rank}: Failed with error: {str(e)}")
                logger.error(traceback.format_exc())

    finally:
        # Clean up DDP after all trials
        cleanup_ddp()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    hyperparameter_search()
