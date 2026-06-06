import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
from torch.nn import MultiheadAttention, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.tensorboard import SummaryWriter
import warnings
from Bio import SeqIO
warnings.filterwarnings("ignore", category=DeprecationWarning)
import copy
import os
from scipy.stats import spearmanr, pearsonr
import torch.distributed as dist
import time
from torch.nn import SmoothL1Loss
import hashlib
import signal
import itertools
import csv
import random
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
import logging # Add this import
# Add DDP imports
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict
# Add distributed initialization function
def init_ddp(rank, world_size):
    """Initialize the distributed training environment"""
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
def setup_logging(rank, output_dir):
    """Set up logging; each rank has its own log file"""
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)
   
    log_file = os.path.join(output_dir, f'rank_{rank}.log')
   
    logger = logging.getLogger(f'rank_{rank}')
    logger.setLevel(logging.INFO)
   
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
   
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
   
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
   
    # Format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
   
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
   
    return logger
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
 
    def forward(self, x):
        batch_size, C, width = x.size()
        query = self.query(x).view(batch_size, -1, width).permute(0, 2, 1) # B X CX(N)
        key = self.key(x).view(batch_size, -1, width) # B X C x (*W*H)
        attention = torch.bmm(query, key) # Batch Matrix Multiplication
        attention = F.softmax(attention, dim=-1) # softmax to make the weights sum to 1
        value = self.value(x).view(batch_size, -1, width) # B X C x (*W*H)
        out = torch.bmm(value, attention.permute(0, 2, 1)) # apply attention to the values
        out = out.view(batch_size, C, -1) # reshape to original shape
        out = self.gamma * out + x # residual connection
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
            SelfAttention(mid_channels), # Add self-attention layer
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        # Add a 1x1 convolution to adjust residual channel dimensions
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
 
    def forward(self, x):
        residual = x
        out = self.main(x)
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        out += residual # Add the residual to the output
        out = nn.ReLU(inplace=True)(out) # Apply ReLU activation
        return out
 
class DoubleConv1D(nn.Module):
    """(convolution => [BN] => ReLU) * 2 with residual connection"""
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
        # Add a 1x1 convolution to adjust residual channel dimensions
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
 
    def forward(self, x):
        residual = x
        out = self.main(x)
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        out += residual # Add the residual to the output
        out = nn.ReLU(inplace=True)(out) # Apply ReLU activation
        return out
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
 
class Up(nn.Module):
    """Upscaling then double conv"""
 
    def __init__(self, in_channels, out_channels, bilinear=False): # Set bilinear to False by default
        super().__init__()
 
        # For one-dimensional data, use nearest or linear interpolation
        self.upsample_mode = 'nearest' if not bilinear else 'linear' # Bilinear interpolation is not used for 1D data
        self.align_corners = None # align_corners is not relevant for nearest mode
 
        # Add a 1x1 convolution to reduce channels if needed
        self.extra_conv = nn.Conv1d(in_channels // 2, in_channels, kernel_size=1)
        self.reduce_conv = nn.Conv1d(in_channels * 2, in_channels, kernel_size=1)
        self.conv = DoubleConv1D(in_channels, out_channels)
 
    def forward(self, x1, x2):
        # Upscale x1 to match the spatial dimensions of x2
        x1 = F.interpolate(x1, size=x2.size()[2:], mode=self.upsample_mode, align_corners=self.align_corners)
        x2 = self.extra_conv(x2)
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
 
        # Use a 1x1 convolution to reduce channels if needed
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
         
        # Initialize the first layer
        layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, stride, padding))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_channels))
        layers.append(activation())
        if dropout_prob > 0.0:
            layers.append(nn.Dropout(p=dropout_prob))
         
        # Add intermediate convolution layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, stride, padding))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(activation())
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(p=dropout_prob))
         
        # Final layer without activation or batch normalization
        layers.append(nn.Conv1d(hidden_channels, out_channels, kernel_size, stride, padding))
         
        self.network = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.network(x)
class ComplexUNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=4, base_channels=64): # Add the base_channels parameter
        super(ComplexUNet1D, self).__init__()
        self.inc = DoubleConv1DWithAttention(in_channels, base_channels * 8) # Set the initial channel count to twice base_channels
        #self.down1 = Down(base_channels * 2, base_channels * 4)
        #self.down2 = Down(base_channels * 4, base_channels * 8)
        self.down3 = Down(base_channels * 8, base_channels * 16)
        self.down4 = Down(base_channels * 16, base_channels * 32)
        self.down5 = Down(base_channels * 32, base_channels * 64)
        self.up1 = Up(base_channels * 64, base_channels * 32)
        self.up2 = Up(base_channels * 32, base_channels * 16)
        self.up3 = Up(base_channels * 16, base_channels * 8)
        #self.up4 = Up(base_channels * 8, base_channels * 4)
        #self.up5 = Up(base_channels * 4, base_channels * 2)
         
        self.outc = OutConv1D(base_channels * 8, out_channels) # Increase the output-layer channel count accordingly
 
    def forward(self, x):
        x1 = self.inc(x)
        #x2 = self.down1(x1)
        #x3 = self.down2(x1)
        x4 = self.down3(x1)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x1)
        #x = self.up4(x, x1)
        #x = self.up5(x, x1)
 
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
        x = x.permute(2, 0, 1) # (width, batch_size, c)
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
        # Pass kernel_size as an argument and use it here
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), # Use LeakyReLU as the activation function
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
    def __init__(self, in_channels, out_channels, n_blocks=4, base_channels=64, num_heads=1, num_layers=2, hidden_dim=128):
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
            nn.LeakyReLU(negative_slope=0.01, inplace=True), # Use LeakyReLU as the activation function
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
        # input shape: [N, C, ...]
        # flat_input shape: [N * ..., C]
        flat_input = input.view(-1, self._embedding_dim)
        dist = (flat_input.unsqueeze(1) - self.weight) ** 2
        _, encoding_indices = (-dist).max(1) # [N * ...]
        # Quantize and unflatten
        quantized = self.weight[encoding_indices][:,-1,:]
        # Compute loss terms
        e_latent_loss = F.mse_loss(quantized.detach(), input)
        q_latent_loss = F.mse_loss(quantized, input.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # Per-example averages
        avg_probs = torch.mean(torch.exp(-dist), dim=1)
        perplexity = torch.exp(torch.mean(torch.log(avg_probs)))
        # Return quantized latent, loss, perplexity, and encoding indices
        return quantized, loss, perplexity, encoding_indices
# Positional Encoding
class PositionalEncoding (nn.Module):
    def __init__(self, d_model, max_len=5000):
        super (PositionalEncoding, self).__init__()
        pe = torch.zeros (max_len, d_model)
        position = torch.arange (0, max_len, dtype=torch.float).unsqueeze (1)
        div_term = torch.exp (torch.arange (0, d_model, 2).float () * -(np.log (10000.0) /d_model))
        pe [:, 0::2] = torch.sin (position * div_term)
        pe [:, 1::2] = torch.cos (position * div_term)
        pe = pe.unsqueeze (0).transpose (0, 1)
        self.register_buffer ('pe', pe)
 
    def forward (self, x):
        x = x + self.pe [:x.size (0), :]
        return x
def generate_kmer_features(sequence, k, num_features):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    kmer_features = torch.zeros(len(kmers), num_features, dtype=torch.float)
     
    for i, kmer in enumerate(kmers):
        # Use a hash function to map each k-mer to a fixed-size integer space
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
    return rna_seq_list
def pad_or_truncate(seq, max_len):
    if len(seq) > max_len:
        return seq[:max_len]
    else:
        return seq + [char_to_idx['EOS']] * (max_len - len(seq)) # Pad with indices
def create_mask(seq, mask_prob=0.2):
    # Ensure seq is an integer array
    mask = np.random.rand(len(seq)) < mask_prob
    mask[0] = mask[-1] = False # Do not mask BOS or EOS
    return mask
def calculate_codon_features(rna_seq):
    codon_to_aa = {
            'TTT': 'codon_F', 'TTC': 'codon_F', 'TTA': 'codon_L', 'TTG': 'codon_L',
            'CTT': 'codon_L', 'CTC': 'codon_L', 'CTA': 'codon_L', 'CTG': 'codon_L',
            'ATT': 'codon_I', 'ATC': 'codon_I', 'ATA': 'codon_I', 'ATG': 'uAUG', # Special handling for the ATG start codon in the transcript
            'GTT': 'codon_V', 'GTC': 'codon_V', 'GTA': 'codon_V', 'GTG': 'codon_V',
            'TCT': 'codon_S', 'TCC': 'codon_S', 'TCA': 'codon_S', 'TCG': 'codon_S',
            'CCT': 'codon_P', 'CCC': 'codon_P', 'CCA': 'codon_P', 'CCG': 'codon_P',
            'ACT': 'codon_T', 'ACC': 'codon_T', 'ACA': 'codon_T', 'ACG': 'codon_T',
            'GCT': 'codon_A', 'GCC': 'codon_A', 'GCA': 'codon_A', 'GCG': 'codon_A',
            'TAT': 'codon_Y', 'TAC': 'codon_Y', 'TAA': 'codon_*', 'TAG': 'codon_*', # Stop codon
            'CAT': 'codon_H', 'CAC': 'codon_H', 'CAA': 'codon_Q', 'CAG': 'codon_Q',
            'AAT': 'codon_N', 'AAC': 'codon_N', 'AAA': 'codon_K', 'AAG': 'codon_K',
            'GAT': 'codon_D', 'GAC': 'codon_D', 'GAA': 'codon_E', 'GAG': 'codon_E',
            'TGT': 'codon_C', 'TGC': 'codon_C', 'TGA': 'codon_*', 'TGG': 'codon_W', # TGA is a stop codon
            'CGT': 'codon_R', 'CGC': 'codon_R', 'CGA': 'codon_R', 'CGG': 'codon_R',
            'AGT': 'codon_S', 'AGC': 'codon_S', 'AGA': 'codon_R', 'AGG': 'codon_R',
            'GGT': 'codon_G', 'GGC': 'codon_G', 'GGA': 'codon_G', 'GGG': 'codon_G'
        }
    codon_count = defaultdict(int)
    stop_codon_count = 0
    start_codon_count = 0
   
    # Iterate over the RNA sequence in groups of three characters, i.e. codons
    for i in range(0, len(rna_seq) - 2, 3):
        codon = rna_seq[i:i+3]
        if codon_to_aa.get(codon) == 'uAUG':
            start_codon_count += 1
        elif codon_to_aa.get(codon) == 'codon_*':
            stop_codon_count += 1
        else:
            codon_count[codon_to_aa.get(codon, 'codon_X')] += 1
    # Calculate codon proportions
    total_codons = sum(codon_count.values())
    codon_features = {key: value / total_codons for key, value in codon_count.items() if key != 'codon_X'}
    codon_features['uAUG'] = start_codon_count
    codon_features['uORF'] = stop_codon_count
   
    return codon_features
# Dataset for RNA sequences
class RNADataset(Dataset):
    def __init__(self, file_name, is_train=True):
        self.rna_sequences = []
        self.labels = []
        self.features = []
        self.feature_cols = [
            'codon_H', 'codon_K', 'codon_I', 'codon_M', 'codon_E','codon_*',
            'codon_F', 'codon_D', 'codon_G', 'codon_T', 'codon_A', 'codon_C', 'codon_Q',
            'codon_S', 'codon_L', 'codon_Y', 'codon_P', 'codon_R', 'codon_V', 'codon_W',
            'codon_N', 'uORF', 'uAUG'
        ]
        for record in SeqIO.parse(file_name, "fasta"):
            description = record.description
            rna_sequence = str(record.seq).upper()
            # Extract labels
            label = 1 if "class_1" in description else 0
            # Compute codon features
            features = calculate_codon_features(rna_sequence)
            feature_vector = [features.get(col, 0.0) for col in self.feature_cols]
            self.features.append(feature_vector)
            self.rna_sequences.append(rna_sequence)
            self.labels.append(label)
    def __len__(self):
        return len(self.rna_sequences)
    def __getitem__(self, idx):
        rna_seq = self.rna_sequences[idx]
        features = self.features[idx]
        kmer_features = generate_kmer_features(rna_seq, k=3, num_features=128)
       
        # Data augmentation: random point mutation
        if random.random() < 0.1:
            mutation_idx = random.randint(0, len(rna_seq) - 1)
            mutated_base = random.choice(['A', 'C', 'G', 'T'])
            while mutated_base == rna_seq[mutation_idx]:
                mutated_base = random.choice(['A', 'C', 'G', 'T'])
            rna_seq = rna_seq[:mutation_idx] + mutated_base + rna_seq[mutation_idx+1:]
        rna_seq_list = rna_to_idx(rna_seq) # Convert the RNA sequence to an index list
        rna_seq_with_special = add_special_tokens(rna_seq_list, char_to_idx) # Pass char_to_idx as an argument
        input_tensor = torch.tensor(rna_seq_with_special, dtype=torch.long)
        return input_tensor, self.labels[idx], kmer_features, torch.tensor(features)
class CNNFeatureExtractor(nn.Module):
    def __init__(self, embed_dim, num_filters, filter_size):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(embed_dim, num_filters, filter_size, padding=filter_size//2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, filter_size, padding=filter_size//2)
       
    def forward(self, x):
        x = x.permute(0, 2, 1)
        # Use Leaky ReLU as the activation function
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01) # The negative_slope parameter can be adjusted
           
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01) # Keep consistency by using the same negative_slope parameter
           
        x = F.avg_pool1d(x, x.size(2))
        x = x.squeeze(2) # Remove the last dimension
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
 
        # Add relative positional encoding to attention
        rel_pos_bias = self.rel_pos_encoding(q, k)
 
        # If src_mask is provided, combine it with rel_pos_bias in some way (if needed)
        # Here we assume we don't need src_mask and use only rel_pos_bias
        attn_output, attn_output_weights = self.self_attn(
            q, k, value=v, key_padding_mask=src_key_padding_mask # Use key_padding_mask instead of attn_mask
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
        """Dynamically compute distance buckets and use a nonlinear function to compute bucket indices."""
        num_buckets = self.max_relative_positions // 4 + 1
        bucket_indices = (torch.abs(relative_positions) / 4).float()
        num_buckets_tensor = torch.tensor(num_buckets, dtype=bucket_indices.dtype, device=bucket_indices.device)
        bucket_indices = torch.log(bucket_indices + 1e-6) / torch.log(num_buckets_tensor)
        bucket_indices = bucket_indices.clamp(0, 1) * (num_buckets - 1) # Normalize and map to the bucket-index range
        return bucket_indices.long()
     
    def forward(self, q, k):
        seq_length = q.size(1)
        relative_positions = torch.arange(seq_length, device=q.device).view(1, seq_length, 1) - torch.arange(seq_length, device=q.device).view(1, 1, seq_length)
        rel_indices = relative_positions.clamp(-self.max_relative_positions + 1, self.max_relative_positions - 1) + self.max_relative_positions
        bucket_indices = self.compute_distance_bucket(rel_indices) # Dynamically compute bucket indices
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
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01) # Adjust the negative_slope parameter as needed
    def forward(self, src, key_padding_mask=None, attn_mask=None):
        N, seq_len, embed_dim = src.size()
        queries = self.queries(src).reshape(N, seq_len, self.num_heads, self.head_dim)
        keys = self.keys(src).reshape(N, seq_len, self.num_heads, self.head_dim)
        values = self.values(src).reshape(N, seq_len, self.num_heads, self.head_dim)
    
        # Scale dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
 
        # Add relative position encoding
        rel_pos_bias = self.rel_pos_encoding(queries, keys)
        query_len, key_len, depth = rel_pos_bias.shape
         
        # Expand rel_pos_bias to include batch size (N) and heads dimensions
        rel_pos_bias = rel_pos_bias.unsqueeze(0) # Add batch dimension
        rel_pos_bias = rel_pos_bias.unsqueeze(0) # Add heads dimension (placed before query_len for clarity)
        rel_pos_bias = rel_pos_bias.expand(N, self.num_heads, query_len, key_len, depth)
         
        # Permute to match energy's dimensions [N, heads, query_len, key_len, depth]
        rel_pos_bias = rel_pos_bias.permute(0, 1, 2, 3, 4)
         
        # Sum along the depth dimension to match energy's shape
        rel_pos_bias = rel_pos_bias.sum(dim=-1)
        energy += rel_pos_bias
 
        if attn_mask is not None:
            energy += attn_mask
        if key_padding_mask is not None:
            # Add batch_size and num_heads dimensions
            key_padding_mask = key_padding_mask.transpose(0, 1) # Transpose back to [batch_size, seq_len]
            seq_len = energy.size(2) # Get sequence length
            key_padding_mask = key_padding_mask.unsqueeze(1).expand(-1, self.num_heads, -1) # [batch_size, num_heads, seq_len]
            key_padding_mask = key_padding_mask.unsqueeze(2).repeat(1, 1, seq_len, 1) # [batch_size, num_heads, seq_len, seq_len]
            # Create a zero tensor with the same shape as energy and use masked_fill_ to set True positions in key_padding_mask to -inf
            energy_mask = torch.zeros_like(energy, device=energy.device)
            energy_mask = energy_mask.masked_fill(key_padding_mask.bool(), float('-inf'))
         
            # Add energy_mask to energy
            energy = energy + energy_mask
                 
 
        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=-1)
        attention = self.dropout(attention)
 
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_dim
        )
 
        out = self.fc_out(out)
        out = self.leakyrelu(out) # Apply LeakyReLU activation
        return out, attention
class EnhancedEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.2, max_relative_positions=64):
        super(EnhancedEncoderLayer, self).__init__()
        self.d_model = d_model # Add this line
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
       
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model) # Extra LayerNorm for the integrated output
 
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        src2 = self.norm1(src)
        attn_output, attn_output_weights = self.self_attn(src2, key_padding_mask=src_key_padding_mask,attn_mask=src_mask)
        src = src + attn_output
       
        # Feed-forward network
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
        self.d_model = d_model # Add this line
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout, max_relative_positions=max_relative_positions)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
         
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model) # Extra LayerNorm
 
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.norm1(tgt)
        tgt2, self_attn_weights = self.self_attn(tgt2, key_padding_mask=None,attn_mask=None)
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
class VAEWithTransformer (nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, num_encoder_layers,
                 num_decoder_layers, num_heads, num_embeddings, commitment_cost, kmer_feature_dim):
        super(VAEWithTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.vector_quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        # Embedding layer
        self.embedding = nn.Embedding (vocab_size, embed_dim)
        self.embed_scale = nn.Parameter(torch.tensor(math.sqrt(embed_dim)))
        self.pos_encoder = PositionalEncoding (embed_dim)
       
        # Encoder
        encoder_layer = EnhancedEncoderLayer(d_model=embed_dim, nhead=num_heads, max_relative_positions=8)
        self.transformer_encoder = EnhancedTransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
 
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, embed_dim)
        self.decoder_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_decoder = PositionalEncoding (embed_dim)
        # Latent space transformation
        self.fc_mu = nn.Linear (embed_dim, latent_dim)
        self.fc_logvar = nn.Linear (embed_dim, latent_dim)
 
        # Decoder
        decoder_layer = EnhancedTransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, max_relative_positions=8)
        self.transformer_decoder = EnhancedTransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        # Output layer
        self.fc_out = nn.Linear (embed_dim, vocab_size)
        self.fc_base_pred = nn.Linear(embed_dim, vocab_size)
        self.cnn_feature_extractor0 = CNNFeatureExtractor(embed_dim, num_filters=latent_dim*2, filter_size=3)
        self.cnn_feature_extractor = CNNFeatureExtractor(embed_dim, num_filters=latent_dim, filter_size=3)
        # Output layer for classification
        self.fc_class = nn.Linear(latent_dim, 2) # Output is binary classification
        self.unet = ComplexTransformerUNet1D(self.embed_dim, self.embed_dim)
        self.kmer_fc = nn.Linear(kmer_feature_dim, self.embed_dim)
        self.fc_z_q_to_decoder = nn.Linear(latent_dim, embed_dim)
        self.feature_embedding = nn.Sequential(
            nn.Linear(23, hidden_dim), # Hidden layer; hidden_dim can be larger than 26
            nn.LeakyReLU(inplace=True), # Replace ReLU with LeakyReLU
            nn.Linear(hidden_dim, embed_dim) # Output layer matching the embedding dimension
        )
        
        self.norm1 = nn.LayerNorm(self.embed_dim)
    def encode(self, src):
        src_mask = self._generate_square_subsequent_mask(len(src)).to(src.device)
        # Embedding and positional encoding
        src = self.embedding(src)* self.embed_scale
        src = self.norm1(src)
        src_key_padding_mask = (src == 0).any (dim=-1).transpose (0, 1) # Create a mask with shape (batch_size, seq_len)
        encoder_output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        encoder_output_avg_pooled = encoder_output.mean(dim=1)
        h = F.relu(self.fc1(encoder_output_avg_pooled))
        mu = self.fc21(h)
        log_var = self.fc22(h)
        return mu, log_var, encoder_output
    def reparameterize (self, mu, logvar):
        std = torch.exp (0.5 * logvar)
        eps = torch.randn_like (std)
        return mu + eps * std
    def decode(self, src, z):
        tgt_seq_len = src.size(1)
        h = F.relu(self.fc3(z))
        h = self.fc4(h).unsqueeze(1).expand(-1, tgt_seq_len, -1)
        emb = self.decoder_embedding(torch.full((h.size(0), tgt_seq_len), 0, dtype=torch.long).to(h.device))
        emb = emb * math.sqrt(self.decoder_embedding.embedding_dim)
        #emb = self.pos_decoder(emb)
        memory = self.transformer_encoder(emb) # Here we use the encoder again as memory for decoder
        output = self.transformer_decoder(emb, memory)
        #emb = self.pos_decoder(emb)
        return output
    def forward (self, src, kmer_features, stacked_features, noise_std=0.0):
        mu, log_var, encoder_output = self.encode(src)
        z = self.reparameterize(mu, log_var)
        z_q, vq_loss, _, _ = self.vector_quantizer(z)
        noisy_encoder_output = encoder_output + noise_std * torch.randn_like(encoder_output)
        denoised_output = self.unet(noisy_encoder_output.permute(0, 2, 1)) # Use permute to match the UNet input format
        decoder_input = denoised_output.permute(0, 2, 1) # Assume denoised_output and z_q_transformed have compatible dimensions
        decoder_input_features = self.cnn_feature_extractor(decoder_input)
        decoder_output = self.decode(src, decoder_input_features)
        logits = self.fc_out(decoder_output)
        kmer_emb = self.kmer_fc(kmer_features)
        kmer_emb_avg_pooled = kmer_emb.mean(dim=1)
        feature_embedded = self.feature_embedding(stacked_features.unsqueeze(1))
        combined_features = torch.cat([denoised_output.permute(0, 2, 1)], dim=1)
        cnn_features = self.cnn_feature_extractor(combined_features)
        class_pred = self.fc_class(cnn_features)
        denoising_loss = F.mse_loss(denoised_output.permute(0, 2, 1), encoder_output)
        return class_pred, logits, mu, log_var, vq_loss,denoising_loss
    def _generate_square_subsequent_mask (self, sz):
        mask = (torch.triu (torch.ones (sz, sz)) == 1).transpose (0, 1)
        mask = mask.float ().masked_fill (mask == 0, float ('-inf')).masked_fill (mask == 1, float (0.0))
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
def loss_function(logits, targets, mu, logvar, vq_loss, denoising_loss,
                  class_preds, class_targets, class_weight=1.0, beta=1.0, denoise_weight=1.0, vq_beta=1.0):
    cross_entropy = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    class_targets = torch.nn.functional.one_hot(class_targets.long(), num_classes=2).float()
    class_loss = F.binary_cross_entropy_with_logits(class_preds, class_targets, reduction='mean')
    return cross_entropy + beta * KLD + class_weight * class_loss + vq_beta * vq_loss + denoise_weight * denoising_loss
def calculate_accuracy(predicted, targets, mask):
    _, predicted_indices = torch.max(predicted, dim=1)
    correct = (predicted_indices == targets) & mask.bool()
    accuracy = correct.float().sum() / mask.sum().item()
    return accuracy.item()
def collate_fn(batch, max_len=150, vocab=None):
    seq_tensors, labels, kmer_features, features = zip(*batch)
   
    # Determine the maximum length, including the retained <EOS>
    max_len_with_eos = max(len(seq) for seq in seq_tensors) if max_len is None else max_len
    max_len_kmer = max(feat.size(0) for feat in kmer_features)
   
    padded_seqs = torch.full((len(seq_tensors), max_len_with_eos), vocab['<PAD>'], dtype=torch.long)
    padded_kmer_features = []
    padded_features = [] # Added: store padded feature tensors
   
    for i, (seq, feat, extra_feat) in enumerate(zip(seq_tensors, kmer_features, features)):
        # Truncate the sequence to the maximum length, excluding <EOS>
        if seq.size(0) > max_len_with_eos:
            seq = seq[:max_len_with_eos]
       
        # Pad the sequence to the maximum length, including <EOS>
        padded_seqs[i, :seq.size(0)] = seq
       
        # Pad k-mer features
        padded_feat = F.pad(feat, (0, 0, 0, max_len_kmer - feat.size(0)), mode='constant', value=0)
        padded_kmer_features.append(padded_feat)
       
        # Pad additional features; no truncation is needed because they have fixed length
        padded_features.append(extra_feat.clone().detach())
   
    # Stack padded tensors into a batch
    stacked_kmer_features = torch.stack(padded_kmer_features)
    stacked_features = torch.stack(padded_features) # Added: stack padded feature tensors
   
    return padded_seqs, torch.tensor(labels, dtype=torch.long), stacked_kmer_features, stacked_features
# Data loading modified to use DistributedSampler
def get_data_loader(dataset, batch_size, vocab=None, shuffle=True, rank=0, world_size=1):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle) if world_size > 1 else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        collate_fn=lambda x: collate_fn(x, max_len=128, vocab=vocab),
        num_workers=4,
        pin_memory=True
    )
def evaluate_model(model, data_loader, device, rank, world_size):
    model.eval()
    all_preds = []
    all_labels = []
    all_class_preds = [] # Store predictions from all batches
    total_loss = 0.0 # Added: store total loss
    num_batches = 0 # Added: record the number of batches
   
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels, stacked_kmer_features, stacked_features = batch
            inputs, labels = inputs.to(device), labels.to(device)
            stacked_kmer_features, stacked_features = stacked_kmer_features.to(device), stacked_features.to(device)
           
            # Get classification predictions
            class_preds, _, _, _, _, _ = model(inputs, stacked_kmer_features, stacked_features)
           
            # Collect predictions from all batches
            all_class_preds.append(class_preds)
           
            # Use argmax to convert logits into class labels
            preds = torch.argmax(class_preds, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            labels = torch.nn.functional.one_hot(labels.long(), num_classes=2).float()
            batch_loss = F.binary_cross_entropy_with_logits(class_preds, labels)
            total_loss += batch_loss
            num_batches += 1
   
    # Merge predictions from all batches
    all_class_preds = torch.cat(all_class_preds, dim=0)
   
    # Compute average loss
    avg_loss = total_loss / num_batches
   
    # Compute evaluation metrics
    acc = accuracy_score(all_labels, all_preds)
   
    # Use only positive-class probabilities to compute AUC
    pos_class_preds = torch.sigmoid(all_class_preds[:, 1]).cpu().numpy() # Assume the positive-class index is 1
    auc = roc_auc_score(all_labels, pos_class_preds)
   
    auprc = average_precision_score(all_labels, pos_class_preds)
    f1 = f1_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
   
    # Each rank prints its own results
    #print(f"Rank {rank}: Local Results - Loss: {avg_loss:.4f}, ACC: {acc:.4f}, AUC: {auc:.4f}, "
    # f"AUPRC: {auprc:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")
   
    return acc, auc, auprc, f1, mcc, avg_loss
def calculate_sequence_lengths(file_name, vocab):
    lengths = []
    with open(file_name, 'r') as file:
        for line in file:
            if not line.startswith('>'):
                rna_sequence = line.strip().upper()
                lengths.append(len(rna_sequence))
     
    return lengths
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, rank=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.rank = rank
        # Store the best metric for the filename
        self.best_metrics = {}
    def __call__(self, val_loss, model, metrics=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_metrics = metrics if metrics else {}
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'Rank {self.rank} - EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_metrics = metrics if metrics else {}
            self.counter = 0
            self.save_checkpoint(val_loss, model)
    def save_checkpoint(self, val_loss, model):
        '''Save the current best model'''
        # Use the metric in the filename
        if self.best_metrics:
            acc = self.best_metrics.get('acc', 0)
            auc = self.best_metrics.get('auc', 0)
            mcc = self.best_metrics.get('mcc', 0)
            auprc = self.best_metrics.get('auprc', 0)
            f1 = self.best_metrics.get('f1', 0)
            fold = self.best_metrics.get('fold', 0)
            rep = self.best_metrics.get('rep', 0)
            filename = f'./Model/best_model_rank_{self.rank}_Fold_{fold}_rep_{rep}_ACC_{acc:.4f}_AUC_{auc:.4f}_MCC_{mcc:.4f}_F1_{f1:.4f}_AUPRC_{auprc:.4f}.pth'
        else:
            filename = f'./Model/best_model_rank_{self.rank}.pth'
           
        torch.save(model.state_dict(), filename)
        if self.verbose:
            print(f'Rank {self.rank} - Test loss improved ({self.best_score:.6f} --> {val_loss:.6f}). Saving model ...')
parser = argparse.ArgumentParser (description='Train VAE with Transformer')
parser.add_argument ('--train_file', type=str, default = './fold_0/train.csv',help='Input CSV file')
parser.add_argument ('--test_file', type=str, default = './fold_0/test.csv',help='Input CSV file')
parser.add_argument ('--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument ('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument ('--learning_rate', type=float, default=0.000001, help='Learning rate')
parser.add_argument ('--embed_dim', type=int, default=128, help='embed dimension size')
parser.add_argument ('--hidden_dim', type=int, default=128, help='Hidden dimension size')
parser.add_argument ('--latent_dim', type=int, default=128, help='Latent space dimension size')
parser.add_argument ('--num_encoder_layers', type=int, default=2, help='Number of layers in encoder')
parser.add_argument ('--num_decoder_layers', type=int, default=2, help='Number of layers in decoder')
parser.add_argument ('--num_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument ('--gpus', type=str, default='0,1', help='GPUs to use')
parser.add_argument('--num_embeddings', type=int, default=32, help='Number of embeddings in the codebook')
parser.add_argument('--commitment_cost', type=float, default=0.1, help='Commitment cost for VQ-VAE')
parser.add_argument('--device', type=str, default="cuda:1", help='Device')
# Add distributed training arguments
parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for logs and models')
args = parser.parse_args ()
print(args)
# Get distributed training environment variables
rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
# Set up logging
logger = setup_logging(rank, args.output_dir)
# Set the GPU device
torch.cuda.set_device(rank)
# Initialize distributed training
init_ddp(rank, world_size)
# Main training loop modified to clear memory after each fold
for j in range(1,5): # 5-fold cross-validation
    logger.info(f'Rank {rank}: Starting fold {j}')
    device = torch.device(f'cuda:{rank}')
    vocab = {'<PAD>': 0,'<SOS>': 1, '<EOS>': 2, '<UNK>': 3,'A': 4, 'T': 5, 'G': 6, 'C': 7, '<CLS>': 8}
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for idx, char in enumerate(vocab)}
    train_file = './fold'+ str(j) +'/train.fa'
    val_file = './fold'+ str(j) +'/dev.fa'
    test_file = './test.fa'
   
    # Load training and validation sets
    train_dataset = RNADataset(train_file, is_train=False)
    val_dataset = RNADataset(val_file, is_train=False)
   
    # Merge training and validation sets
    combined_train_dataset = ConcatDataset([train_dataset, val_dataset])
   
    test_dataset = RNADataset(test_file, is_train=False)
    # Use distributed data loaders
    train_loader = get_data_loader(combined_train_dataset, batch_size=args.batch_size,
                                 vocab=vocab, shuffle=True, rank=rank, world_size=world_size)
    test_loader = get_data_loader(test_dataset, batch_size=args.batch_size,
                                vocab=vocab, shuffle=False, rank=rank, world_size=world_size)
    vocab_size = len (vocab)
    initial_noise_std = 0.01
    max_noise_std = 0.1
    # Run multiple repeated experiments within each fold
    for i in range(5):
        logger.info(f"Rank {rank}: Starting repetition {i} for fold {j}")
       
        # Clear the GPU cache
        torch.cuda.empty_cache()
       
        noise_schedule = cosine_noise_schedule(initial_noise_std, max_noise_std, args.epochs)
    
        model = VAEWithTransformer(vocab_size, args.embed_dim, args.hidden_dim, args.latent_dim,
                        args.num_encoder_layers, args.num_decoder_layers, args.num_heads,
                        args.num_embeddings, args.commitment_cost, kmer_feature_dim=128)
        
        # Load pretrained model
        pretrained_path = './Model/best_model_rank_2_Fold_0_rep_2_ACC_0.8468_AUC_0.8783_MCC_0.6297_F1_0.7377_AUPRC_0.7700.pth'
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    
        model.to(device)
       
        # Wrap the model with DDP and set find_unused_parameters=True
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
       
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5, amsgrad=True)
        #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.1)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
        early_stopping = EarlyStopping(patience=20, verbose=True, rank=rank)
    
        best_MCC = float('-inf') # Initialize the best MCC to negative infinity
        best_model_state = None # Initialize the best model state as None
    
        for epoch in range(args.epochs):
            start_time = time.time()
            model.train()
            total_loss = 0.0
           
            # Set the epoch for the sampler so data are shuffled differently each epoch
            if world_size > 1:
                train_loader.sampler.set_epoch(epoch)
           
            # Show the progress bar only on rank 0
            #if rank == 0:
            #    batch_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
            #else:
            #    batch_iterator = train_loader
               
            for batch in train_loader:
                src, labels, stacked_kmer_features, stacked_features = batch
                src, labels, stacked_kmer_features, stacked_features = (
                    src.to(device), labels.to(device), stacked_kmer_features.to(device), stacked_features.to(device)
                )
                targets = src
                class_targets = torch.tensor([1 if label == 1 else 0 for label in labels], dtype=torch.float).to(device)
                noise_std = noise_schedule(epoch)
    
                class_preds, logits, mu, logvar, vq_loss, denoising_loss = model(
                    src, stacked_kmer_features, stacked_features, noise_std=noise_std
                )
                loss = loss_function(
                    logits, targets, mu, logvar, vq_loss, denoising_loss, class_preds, class_targets,
                    class_weight=1.0, beta=1.0, denoise_weight=1.0, vq_beta=1.0
                )
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                total_loss += loss.item()
    
            lr_scheduler.step()
           
            for param_group in optimizer.param_groups:
                if param_group['lr'] < 0.000001:
                    param_group['lr'] = 0.000001
            end_time = time.time()
            avg_loss = total_loss / len(train_loader)
           
            # Each rank prints its own training loss
            #logger.info(f"Rank {rank} - Fold {j} Rep {i} Epoch {epoch+1}: Training Loss: {avg_loss:.4f}, "
            # f"LR: {optimizer.param_groups[0]['lr']:.6f}, Time: {end_time-start_time:.2f}s")
    
            # Evaluate the model independently on each rank
            acc, auc, auprc, f1, mcc, class_loss = evaluate_model(
                model, test_loader, device, rank, world_size
            )
    
            logger.info(f'Rank {rank} - Fold {j} Rep {i} Epoch {epoch+1}/{args.epochs} | '
                      f'Loss: {class_loss:.4f} | ACC: {acc:.4f} | AUC: {auc:.4f} | AUPRC: {auprc:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}')
               
            # Update the best model state when the metric improves
            if mcc > best_MCC:
                best_MCC = mcc
                best_model_state = copy.deepcopy(model.module.state_dict())
                #logger.info(f"Rank {rank} - New best MCC: {best_MCC:.4f}")
           
            # Update the early-stopping call and pass the metric
            metrics = {'acc': acc, 'auc': auc, 'mcc': mcc, 'auprc': auprc,'f1': f1,'fold': j,'rep': i}
            early_stopping(-mcc, model.module, metrics)
            if early_stopping.early_stop:
                logger.info(f"Rank {rank} - Early stopping triggered")
                break
        
        # Each rank saves its own best model
        if best_model_state is not None:
            best_model = VAEWithTransformer(vocab_size, args.embed_dim, args.hidden_dim, args.latent_dim,
                                            args.num_encoder_layers, args.num_decoder_layers, args.num_heads,
                                            args.num_embeddings, args.commitment_cost, kmer_feature_dim=128)
            best_model.load_state_dict(best_model_state)
           
            # Save the model filename using the requested format
            model_filename = f'./Model/rank_{rank}_fold_{j}_rep_{i}__ACC_{acc:.4f}_AUC_{auc:.4f}_MCC_{mcc:.4f}_F1_{f1:.4f}_AUPRC_{auprc:.4f}.pth'
            torch.save(best_model.state_dict(), model_filename)
            #logger.info(f'Rank {rank} - Saved best model for fold {j} rep {i} with ACC: {acc:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}, AUPRC: {auprc:.4f}')
        # Clean up resources for the current repeated experiment
        del model, optimizer, lr_scheduler, early_stopping
        torch.cuda.empty_cache()
       
        #logger.info(f"Rank {rank} - Completed repetition {i} for fold {j}")
    # Clean up datasets and data loaders for the current fold
    del train_dataset, val_dataset, test_dataset, combined_train_dataset
    del train_loader, test_loader
    torch.cuda.empty_cache()
   
    #logger.info(f"Rank {rank} - Completed fold {j}")
# Clean up the distributed training environment
dist.destroy_process_group()
#logger.info(f"Rank {rank}: Training completed")