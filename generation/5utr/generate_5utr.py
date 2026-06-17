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
import torch.distributed as dist
import time
from torch.nn import SmoothL1Loss
from torch.cuda.amp import autocast, GradScaler
import hashlib
import signal
import itertools
import csv
import random
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
import pandas as pd
from collections import Counter
from Bio.Seq import Seq
from ViennaRNA import RNA  # For MFE computation
import re  # For cleaning sequences
import multiprocessing
from multiprocessing import Pool, cpu_count
import copy

def compute_features(utr):
    utr = str(utr).upper()  # 确保大写DNA
    length = len(utr)
    uaug = utr.count('ATG')
    
    # aa frequency
    total_codons = length // 3
    aa_map = {
        'H': 'codon_H', 'K': 'codon_K', 'I': 'codon_I', 'M': 'codon_M', 'E': 'codon_E',
        'F': 'codon_F', 'D': 'codon_D', 'G': 'codon_G', 'T': 'codon_T', 'A': 'codon_A',
        'C': 'codon_C', 'Q': 'codon_Q', 'S': 'codon_S', 'L': 'codon_L', 'Y': 'codon_Y',
        '*': 'codon_*', 'P': 'codon_P', 'R': 'codon_R', 'V': 'codon_V', 'W': 'codon_W', 'N': 'codon_N'
    }
    if total_codons > 0:
        try:
            seq_aa = Seq(utr[:total_codons * 3])  # 只取完整密码子部分
            protein = seq_aa.translate(to_stop=False)
            aas = list(str(protein))
            aa_count = Counter(aas)
            aa_freq = {aa_map[aa]: aa_count.get(aa, 0) / total_codons for aa in aa_map}
        except Exception:
            aa_freq = {v: 0 for v in aa_map.values()}
    else:
        aa_freq = {v: 0 for v in aa_map.values()}
    
    # uORF prediction: scan all positions for ATG, check in-frame stop
    seq = Seq(utr)
    uorfs = []
    for i in range(length - 2):
        if str(seq[i:i+3]) == 'ATG':
            stop_found = False
            stop_pos = None
            for j in range(i + 3, length, 3):
                codon = str(seq[j:j+3])
                if len(codon) == 3 and codon in ['TAA', 'TAG', 'TGA']:
                    stop_found = True
                    stop_pos = j + 3
                    break
            uorf_length = length - i if not stop_found else (stop_pos - i)
            uorfs.append({'has_stop': stop_found, 'length': uorf_length})
    uorf = sum(1 for u in uorfs if u['has_stop'] and u['length'] >= 9)
    
    # 核苷酸计数
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
    
    # 组合结果
    result = aa_freq
    result['uAUG'] = uaug
    result['uORF'] = uorf
    result['CGratio'] = cgratio
    result['CGperc'] = cgperc
    result['utrlen_m80'] = utrlen_m80
    result['ATratio'] = atratio
    return result

def compute_utr_mfe(dna_sequence, temperature=37.0):
    """
    计算 UTR DNA 序列的 RNA MFE 使用 ViennaRNA。
    """
    # 转换为 RNA 序列 (T -> U)，并清洗
    rna_sequence = re.sub(r'[^AUGC]', '', dna_sequence.upper()).replace('T', 'U')
    
    if len(rna_sequence) < 1:
        raise ValueError("序列不能为空")
    
    # 创建折叠化合物
    fc = RNA.fold_compound(rna_sequence)
    
    # 计算 MFE 和结构
    (structure, mfe) = fc.mfe()
    
    return mfe, structure

class SelfAttention(nn.Module):  
    def __init__(self, in_channels):  
        super(SelfAttention, self).__init__()  
        self.query = nn.Conv1d(in_channels, in_channels // 8, 1)  
        self.key = nn.Conv1d(in_channels, in_channels // 8, 1)  
        self.value = nn.Conv1d(in_channels, in_channels, 1)  
        self.gamma = nn.Parameter(torch.zeros(1))  
  
    def forward(self, x):  
        batch_size, C, width = x.size()  
        query = self.query(x).view(batch_size, -1, width).permute(0, 2, 1)  # B X CX(N)  
        key = self.key(x).view(batch_size, -1, width)  # B X C x (*W*H)  
        attention = torch.bmm(query, key)  # Batch Matrix Multiplication  
        attention = F.softmax(attention, dim=-1)  # softmax to make the weights sum to 1  
        value = self.value(x).view(batch_size, -1, width)  # B X C x (*W*H)  
        out = torch.bmm(value, attention.permute(0, 2, 1))  # apply attention to the values  
        out = out.view(batch_size, C, -1)  # reshape to original shape  
        out = self.gamma * out + x  # residual connection  
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
            SelfAttention(mid_channels),  # 添加自注意力层  
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),  
            nn.BatchNorm1d(out_channels)  
        )  
        # 添加1x1卷积来调整残差的通道数  
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None  
  
    def forward(self, x):  
        residual = x  
        out = self.main(x)  
        if self.residual_conv is not None:  
            residual = self.residual_conv(residual)  
        out += residual  # 将残差加到输出上  
        out = nn.ReLU(inplace=True)(out)  # 应用ReLU激活函数  
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
        # 添加1x1卷积来调整残差的通道数  
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None  
  
    def forward(self, x):  
        residual = x  
        out = self.main(x)  
        if self.residual_conv is not None:  
            residual = self.residual_conv(residual)  
        out += residual  # 将残差加到输出上  
        out = nn.ReLU(inplace=True)(out)  # 应用ReLU激活函数  
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
  
    def __init__(self, in_channels, out_channels, bilinear=False):  # 注意这里将bilinear默认为False  
        super().__init__()  
  
        # 对于一维数据，我们总是使用nearest或linear插值  
        self.upsample_mode = 'nearest' if not bilinear else 'linear'  # 但实际上，对于1D，我们不会使用bilinear  
        self.align_corners = None  # 对于nearest模式，align_corners不重要  
  
        # 添加一个1x1卷积来减少通道数（如果需要的话）  
        self.extra_conv = nn.Conv1d(in_channels // 2, in_channels, kernel_size=1)  
        self.reduce_conv = nn.Conv1d(in_channels * 2, in_channels, kernel_size=1)  
        self.conv = DoubleConv1D(in_channels, out_channels) 
  
    def forward(self, x1, x2):  
        # Upscale x1 to match the spatial dimensions of x2  
        x1 = F.interpolate(x1, size=x2.size()[2:], mode=self.upsample_mode, align_corners=self.align_corners)  
        x2 = self.extra_conv(x2)  
        # Concatenate along the channel dimension  
        x = torch.cat([x2, x1], dim=1)  
  
        # 使用1x1卷积减少通道数（如果需要的话）  
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
          
        # 初始化第一层  
        layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, stride, padding))  
        if use_batchnorm:  
            layers.append(nn.BatchNorm1d(hidden_channels))  
        layers.append(activation())  
        if dropout_prob > 0.0:  
            layers.append(nn.Dropout(p=dropout_prob))  
          
        # 添加中间的卷积层  
        for _ in range(num_layers - 2):  
            layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, stride, padding))  
            if use_batchnorm:  
                layers.append(nn.BatchNorm1d(hidden_channels))  
            layers.append(activation())  
            if dropout_prob > 0.0:  
                layers.append(nn.Dropout(p=dropout_prob))  
          
        # 最后一层，不包含激活函数和批归一化  
        layers.append(nn.Conv1d(hidden_channels, out_channels, kernel_size, stride, padding))  
          
        self.network = nn.Sequential(*layers)  
  
    def forward(self, x):  
        return self.network(x)  

class ComplexUNet1D(nn.Module):  
    def __init__(self, in_channels, out_channels, n_blocks=4, base_channels=64):  # 增加base_channels参数  
        super(ComplexUNet1D, self).__init__()  
        self.inc = DoubleConv1DWithAttention(in_channels, base_channels * 8)  # 将初始通道数设置为base_channels的2倍  
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
          
        self.outc = OutConv1D(base_channels * 8, out_channels)  # 输出层通道数也相应增加  
  
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
        x = x.permute(2, 0, 1)  # (width, batch_size, c)  
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
        # 将 kernel_size 作为一个参数传入，并在这里使用它  
        self.main = nn.Sequential(    
            nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),    
            nn.BatchNorm1d(hidden_dim),    
            nn.LeakyReLU(negative_slope=0.01, inplace=True),  # 使用LeakyReLU作为激活函数    
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
            nn.LeakyReLU(negative_slope=0.01, inplace=True),  # 使用LeakyReLU作为激活函数  
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
        _, encoding_indices = (-dist).max(1)  # [N * ...]  

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
        # 使用哈希函数将k-mer映射到一个固定大小的整数  
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
    return  [char_to_idx['<CLS>']]  + rna_seq_list 
  
def pad_or_truncate(seq, max_len):  
    if len(seq) > max_len:  
        return seq[:max_len]  
    else:  
        return seq + [char_to_idx['EOS']] * (max_len - len(seq))  # 使用索引填充
def create_mask(seq, mask_prob=0.2):  
    # 确保seq是整数数组  
    mask = np.random.rand(len(seq)) < mask_prob  
    mask[0] = mask[-1] = False  # 不掩盖 BOS 和 EOS  
    return mask  



# Dataset for RNA sequences  
class RNADataset(Dataset):  
    def __init__(self, file_name, vocab, is_train=True):  
        self.rna_sequences = []  
        self.mrl_values = []  
        self.vocab = vocab  
        self.is_train = is_train
        self.feature_cols = [  
            'codon_H', 'codon_K', 'codon_I', 'uAUG', 'CGratio', 'codon_M', 'codon_E',  
            'codon_F', 'codon_D', 'codon_G', 'codon_T', 'codon_A', 'codon_C', 'codon_Q',  
            'codon_S', 'codon_L', 'codon_Y', 'codon_P', 'codon_R', 'codon_*','codon_V', 'codon_W',  
            'codon_N', 'uORF', 'CGperc', 'utrlen_m80', 'ATratio'  
        ]  
        '''
        self.feature_cols = [  
            'codon_H', 'codon_K', 'codon_I', 'codon_M', 'codon_E', 'codon_*', 'codon_N',
            'codon_F', 'codon_D', 'codon_G', 'codon_T', 'codon_A', 'codon_C', 'codon_Q',  
            'codon_S', 'codon_L', 'codon_Y', 'codon_P', 'codon_R', 'codon_V', 'codon_W',  
            'uORF', 'uAUG'
        ] 
        '''

        with open(file_name, 'r', encoding='latin1') as file:  
            reader = csv.DictReader(file)  
            for row in reader:  
                rna_sequence = row['utr'].strip().upper()  
                mrl_value = float(row['rl'])
                mrl_value = round(mrl_value, 2)  
                self.rna_sequences.append(rna_sequence)  
                self.mrl_values.append(mrl_value)  
        #self.indices = [torch.tensor([vocab[base] for base in seq], dtype=torch.long) for seq in self.rna_sequences]  

    def __len__(self):  
        return len (self.rna_sequences)  
    
    def __getitem__(self, idx):  
        rna_seq = self.rna_sequences[idx]  
        
        # 数据增强：随机点突变  
        if self.is_train and random.random() < 0.1:  # 以10%的概率进行突变  
            mutation_idx = random.randint(0, len(rna_seq) - 1)  
            mutated_base = random.choice(['A', 'C', 'G', 'T'])  
            while mutated_base == rna_seq[mutation_idx]:  # 确保突变是有效的
                mutated_base = random.choice(['A', 'C', 'G', 'T'])  
            rna_seq = rna_seq[:mutation_idx] + mutated_base + rna_seq[mutation_idx+1:]  
        
        # Compute features on (possibly mutated) sequence
        feature_dict = compute_features(rna_seq)
        features_list = [feature_dict.get(col, 0.0) for col in self.feature_cols]
        
        kmer_features = generate_kmer_features(rna_seq, k=3, num_features=128)  
        
        rna_seq_list = rna_to_idx(rna_seq)  # 将RNA序列转换为索引列表  
        rna_seq_with_special = add_special_tokens(rna_seq_list, char_to_idx)  # 传递char_to_idx作为参数  
  
        input_tensor = torch.tensor(rna_seq_with_special, dtype=torch.long)  
  
        return input_tensor, self.mrl_values[idx], kmer_features, torch.tensor(features_list, dtype=torch.float)



class CNNFeatureExtractor(nn.Module):    
    def __init__(self, embed_dim, num_filters, filter_size):    
        super(CNNFeatureExtractor, self).__init__()    
        self.conv1 = nn.Conv1d(embed_dim, num_filters, filter_size, padding=filter_size//2)    
        self.conv2 = nn.Conv1d(num_filters, num_filters, filter_size, padding=filter_size//2)    
        
    def forward(self, x):    
        x = x.permute(0, 2, 1)    
        # 使用Leaky ReLU作为激活函数  
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)  # 可以调整negative_slope参数  
            
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)  # 保持一致性，使用相同的negative_slope参数  
            
        x = F.avg_pool1d(x, x.size(2))    
        x = x.squeeze(2)  # 移除最后一个维度    
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
        #print(f"Input src shape: {src.shape}") 
        src2 = self.norm1(src)  
        q, k, v = src2, src2, src2  
  
        # Add relative positional encoding to attention  
        rel_pos_bias = self.rel_pos_encoding(q, k)  
  
        # If src_mask is provided, combine it with rel_pos_bias in some way (if needed)  
        # Here we assume we don't need src_mask and use only rel_pos_bias  
        attn_output, attn_output_weights = self.self_attn(  
            q, k, value=v, key_padding_mask=src_key_padding_mask  # 使用 key_padding_mask 而不是 attn_mask  
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
        """动态计算距离桶，并使用非线性函数计算桶索引。"""  
        num_buckets = self.max_relative_positions // 4 + 1  
        bucket_indices = (torch.abs(relative_positions) / 4).float()  
        num_buckets_tensor = torch.tensor(num_buckets, dtype=bucket_indices.dtype, device=bucket_indices.device)  
        bucket_indices = torch.log(bucket_indices + 1e-6) / torch.log(num_buckets_tensor)
        bucket_indices = bucket_indices.clamp(0, 1) * (num_buckets - 1)  # 归一化并映射到桶索引范围  
        return bucket_indices.long()  
      
    def forward(self, q, k):  
        seq_length = q.size(1)  
        relative_positions = torch.arange(seq_length, device=q.device).view(1, seq_length, 1) - torch.arange(seq_length, device=q.device).view(1, 1, seq_length)  
        rel_indices = relative_positions.clamp(-self.max_relative_positions + 1, self.max_relative_positions - 1) + self.max_relative_positions  
        bucket_indices = self.compute_distance_bucket(rel_indices)  # 动态计算桶索引  
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
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)  # 您可以根据需要调整 negative_slope 参数  



    def forward(self, src, key_padding_mask=None, attn_mask=None):  
        N, seq_len, embed_dim = src.size()  

  
        queries = self.queries(src).reshape(N, seq_len, self.num_heads, self.head_dim)  
        keys = self.keys(src).reshape(N, seq_len, self.num_heads, self.head_dim)  
        values = self.values(src).reshape(N, seq_len, self.num_heads, self.head_dim)  
     
        # Scale dot-product attention  
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  
  
        # Add relative position encoding  
        # Add relative position encoding  
        rel_pos_bias = self.rel_pos_encoding(queries, keys)  
        query_len, key_len, depth = rel_pos_bias.shape  
          
        # Expand rel_pos_bias to include batch size (N) and heads dimensions  
        rel_pos_bias = rel_pos_bias.unsqueeze(0)  # Add batch dimension  
        rel_pos_bias = rel_pos_bias.unsqueeze(0)  # Add heads dimension (placed before query_len for clarity)  
        rel_pos_bias = rel_pos_bias.expand(N, self.num_heads, query_len, key_len, depth)  
          
        # Permute to match energy's dimensions [N, heads, query_len, key_len, depth]  
        rel_pos_bias = rel_pos_bias.permute(0, 1, 2, 3, 4)  
          
        # Sum along the depth dimension to match energy's shape  
        rel_pos_bias = rel_pos_bias.sum(dim=-1)  
        energy += rel_pos_bias  
  
        if attn_mask is not None:  
            energy += attn_mask  
        if key_padding_mask is not None:  
            # 增加 batch_size 和 num_heads 维度  
            key_padding_mask = key_padding_mask.transpose(0, 1)  # 转置回 [batch_size, seq_len]  
            seq_len = energy.size(2)  # 获取序列长度  
            key_padding_mask = key_padding_mask.unsqueeze(1).expand(-1, self.num_heads, -1)  # [batch_size, num_heads, seq_len]  
            key_padding_mask = key_padding_mask.unsqueeze(2).repeat(1, 1, seq_len, 1)  # [batch_size, num_heads, seq_len, seq_len]
            # 创建一个与 energy 形状相同但全为 0 的张量，并使用 masked_fill_ 将 key_padding_mask 中的 True 位置设置为 -inf  
            energy_mask = torch.zeros_like(energy, device=energy.device)  
            energy_mask = energy_mask.masked_fill(key_padding_mask.bool(), float('-inf'))  
          
            # 将 energy_mask 加到 energy 上  
            energy = energy + energy_mask
                  
  
        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=-1)  
        attention = self.dropout(attention)  
  
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(  
            N, query_len, self.embed_dim  
        )  
  
        out = self.fc_out(out)  
        out = self.leakyrelu(out)  # 应用 LeakyReLU 激活函数  

        return out, attention 

class EnhancedEncoderLayer(nn.Module):  
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.2, max_relative_positions=64):  
        super(EnhancedEncoderLayer, self).__init__()
        self.d_model = d_model  # 添加这一行  
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)  
        self.linear1 = nn.Linear(d_model, dim_feedforward)  
        self.dropout = nn.Dropout(dropout)  
        self.linear2 = nn.Linear(dim_feedforward, d_model)  
        
        self.norm1 = nn.LayerNorm(d_model)  
        self.norm2 = nn.LayerNorm(d_model)  
        self.norm3 = nn.LayerNorm(d_model)  # 额外的LayerNorm用于整合后的输出  
  
    def forward(self, src, src_mask=None, src_key_padding_mask=None):  
        # 自注意力  
        src2 = self.norm1(src)  
        attn_output, attn_output_weights = self.self_attn(src2, key_padding_mask=src_key_padding_mask,attn_mask=src_mask)  
        src = src + attn_output  
        
        # 前馈网络  
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
        self.d_model = d_model  # 添加这一行  

        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout, max_relative_positions=max_relative_positions)  
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)  
        self.linear1 = nn.Linear(d_model, dim_feedforward)  
        self.dropout = nn.Dropout(dropout)  
        self.linear2 = nn.Linear(dim_feedforward, d_model)  
          
        self.norm1 = nn.LayerNorm(d_model)  
        self.norm2 = nn.LayerNorm(d_model)  
        self.norm3 = nn.LayerNorm(d_model)  # 额外的LayerNorm  
  
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




# Transformer VAE  
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
        
        # SS预测模块  
        #self.ss_lstm = nn.LSTM(self.embed_dim, hidden_size=64, num_layers=1, batch_first=True)  
  
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

        #self.transformer_decoder = nn.TransformerDecoder(  
        #    nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads), num_layers=num_decoder_layers)  
  
        # Additional FC layer between encoder and decoder  
        self.fc_latent_to_hidden = nn.Linear (latent_dim, embed_dim)  
        
        # Output layer  
        self.fc_out = nn.Linear (embed_dim, vocab_size)
        self.fc_base_pred = nn.Linear(embed_dim, vocab_size)
        self.cnn_feature_extractor0 = CNNFeatureExtractor(embed_dim, num_filters=latent_dim*2, filter_size=3)  
        self.cnn_feature_extractor = CNNFeatureExtractor(embed_dim, num_filters=latent_dim, filter_size=3)  



        #self.fc_mfe_pred = AdvancedRegressionHead(latent_dim, hidden_dim=reg_head_hidden_dim,num_blocks=reg_head_num_blocks, stride = resnet_stride)  
        self.fc_mfe_pred = nn.Linear(latent_dim, 1)
        #self.unet = UNet1D(self.embed_dim, self.embed_dim)
        #self.unet = ComplexUNet1D(self.embed_dim, self.embed_dim)
        self.unet = ComplexTransformerUNet1D(self.embed_dim, self.embed_dim)
        #self.denoiser = DenoisingAutoencoder(self.embed_dim * encoder_output.size(1), self.hidden_dim)  

        self.kmer_fc = nn.Linear(kmer_feature_dim, self.embed_dim)
        self.fc_z_q_to_decoder = nn.Linear(latent_dim, embed_dim)

        #self.feature_embedding = nn.Linear(26, self.embed_dim)  
        self.feature_embedding = nn.Sequential(  
            nn.Linear(27, hidden_dim),  # 隐藏层，hidden_dim 可以是一个比 26 更大的数  
            nn.LeakyReLU(inplace=True),  # 将 ReLU 替换为 LeakyReLU
            nn.Linear(hidden_dim, embed_dim)  # 输出层，匹配嵌入维度  
        )
         
        # 对于 decoder_feature_extractor 部分
        self.decoder_feature_extractor = nn.Sequential(  
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),  
            nn.BatchNorm1d(latent_dim),  
            nn.LeakyReLU(inplace=True),  # 将 ReLU 替换为 LeakyReLU
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),  
            nn.BatchNorm1d(latent_dim),  
            nn.LeakyReLU(inplace=True)  # 将 ReLU 替换为 LeakyReLU
        )

        #self.fc_mfe_pred = AttentionLayer(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(self.embed_dim)  


    def encode(self, src):  
        src_mask = self._generate_square_subsequent_mask(len(src)).to(src.device)  
        # Embedding and positional encoding  
        # 使用可学习的缩放因子来缩放嵌入向量  
        #src = self.embedding(src)  * self.embed_scale  
        src = self.embedding(src)* self.embed_scale
        src = self.norm1(src)

        src_key_padding_mask = (src == 0).any (dim=-1).transpose (0, 1)  # 创建一个形状为 (batch_size, seq_len) 的 mask  
        encoder_output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)  
        encoder_output_avg_pooled = encoder_output.mean(dim=1)

        #mu, logvar = self.fc_mu(encoder_output_avg_pooled), self.fc_logvar(encoder_output_avg_pooled)  
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
        memory = self.transformer_encoder(emb)  # Here we use the encoder again as memory for decoder  
        output = self.transformer_decoder(emb, memory)  
        #emb = self.pos_decoder(emb)  

        return output

    def forward (self, src, kmer_features, stacked_features, noise_std=0.0):  
        mu, log_var, encoder_output = self.encode(src)  
        z = self.reparameterize(mu, log_var)  
        z_q, vq_loss, _, _ = self.vector_quantizer(z) 

        #z_q_transformed = self.fc_z_q_to_decoder(z_q)
        #z_q_transformed_expanded = z_q_transformed.unsqueeze(1).repeat(1,  src.size(1), 1)  # 将z_q_transformed扩展到与tgt相同的序列长度  
        noisy_encoder_output = encoder_output + noise_std * torch.randn_like(encoder_output)  

        denoised_output = self.unet(noisy_encoder_output.permute(0, 2, 1))  # 注意permute以匹配UNet输入  

        decoder_input = denoised_output.permute(0, 2, 1)  # 这里假设denoised_output和z_q_transformed的维度是兼容的  

        #decoder_input_features = self.resnet_feature_extractor(decoder_input)  
        decoder_input_features = self.cnn_feature_extractor(decoder_input)  


        decoder_output = self.decode(src, decoder_input_features)  

        #decoder_features = self.decoder_feature_extractor(decoder_output)  
  
        logits = self.fc_out(decoder_output)  


        #logits = self.fc_out(decoder_output)  

        kmer_emb = self.kmer_fc(kmer_features)
        kmer_emb_avg_pooled = kmer_emb.mean(dim=1) 
        feature_embedded = self.feature_embedding(stacked_features.unsqueeze(1))


        combined_features = torch.cat([denoised_output.permute(0, 2, 1),feature_embedded,kmer_emb], dim=1)  
        cnn_features = self.cnn_feature_extractor(combined_features)

        mfe_pred = self.fc_mfe_pred(cnn_features)

        denoising_loss = F.mse_loss(denoised_output.permute(0, 2, 1), encoder_output)  

        #z_q_transformed = self.fc_z_q_to_decoder(z_q)

        #base_pred_logits = self.fc_base_pred(denoised_output.permute(0, 2, 1))
        #base_pred_logits = self.fc_base_pred(denoised_output.permute(0, 2, 1))


        return logits, mu, log_var, mfe_pred,vq_loss,denoising_loss

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


def loss_function(logits, targets, mu, logvar, vq_loss, denoising_loss, mfe_preds,
 mfe_targets, vocab, beta=1.0, mask_weight=1.0, mfe_weight=1.0, denoise_weight=1.0,vq_beta=1.0):  
    # VAE相关的损失保持不变  

    cross_entropy = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')  
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  
    mfe_loss = F.mse_loss(mfe_preds.squeeze().unsqueeze(0), mfe_targets.squeeze().unsqueeze(0))

    #print(mfe_loss)
    #print(cross_entropy + beta * KLD + mask_weight * base_pred_loss + mfe_weight * mfe_loss + vq_beta * vq_loss + denoise_weight * denoising_loss)
    #toal_loss = cross_entropy + beta * KLD + mask_weight * base_pred_loss + mfe_weight * mfe_loss + vq_beta * vq_loss + denoise_weight * denoising_loss
    return cross_entropy + beta * KLD + mfe_weight * mfe_loss + vq_beta * vq_loss + denoise_weight * denoising_loss,  mfe_loss, denoising_loss


def calculate_accuracy(predicted, targets, mask):    

    _, predicted_indices = torch.max(predicted, dim=1)    

    correct = (predicted_indices == targets) & mask.bool()

    accuracy = correct.float().sum() / mask.sum().item()
    return accuracy.item()




def collate_fn(batch, max_len=150, vocab=None):
    seq_tensors, mfe_values, kmer_features, features = zip(*batch)
    
    # 确定最大长度，包括保留的<EOS>

    max_len_with_eos = max(len(seq) for seq in seq_tensors) if max_len is None else max_len  

    max_len_without_eos = max_len_with_eos - 1  # 减去1，因为我们要保留<EOS>
    max_len_kmer = max(feat.size(0) for feat in kmer_features)
    
    padded_seqs = torch.full((len(seq_tensors), max_len_with_eos), vocab['<PAD>'], dtype=torch.long)
    masked_seqs = []
    padded_kmer_features = []
    padded_features = []  # 新增：用于存储填充后的特征张量
    
    for i, (seq, feat, extra_feat) in enumerate(zip(seq_tensors, kmer_features, features)):
        # 截断序列到最大长度（不包括<EOS>）
        if seq.size(0) > max_len_with_eos:
            seq = seq[:max_len_with_eos]
        
        # 在序列末尾添加<EOS>
        #seq = torch.cat([seq, torch.tensor([vocab['<EOS>']], dtype=torch.long)])
        
        # 填充序列到最大长度（包括<EOS>）
        padded_seqs[i, :seq.size(0)] = seq
        
        # 填充k-mer特征
        padded_feat = F.pad(feat, (0, 0, 0, max_len_kmer - feat.size(0)), mode='constant', value=0)
        padded_kmer_features.append(padded_feat)
        
        # 填充额外特征（这里不需要截断，因为它们是固定长度的）
        padded_features.append(extra_feat.clone().detach())
    
    # 将填充后的张量堆叠成batch
    stacked_kmer_features = torch.stack(padded_kmer_features)
    stacked_features = torch.stack(padded_features)  # 新增：堆叠填充后的特征张量
    
    return padded_seqs, torch.tensor(mfe_values, dtype=torch.float), stacked_kmer_features, stacked_features

# Data loading  
def get_data_loader(dataset, batch_size, vocab=None):  
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if torch.distributed.is_initialized() else None  
    return DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,  
                      collate_fn=lambda x: collate_fn(x, max_len=64, vocab=vocab))


def evaluate_model (model, data_loader, vocab, device):  
    model.eval ()  
    total_correct = 0  
    total_masked = 0  
    predicted_mfe = []  
    true_mfe = []  
    mse_loss_sum = 0  # 初始化 MSE 损失和  
    num_samples = 0  # 初始化样本数量  
    noise_std = 0.0  

    with torch.no_grad ():  
        for batch in data_loader:  
            src, mfe_targets, stacked_kmer_features, stacked_features = batch  
            src, mfe_targets, stacked_kmer_features, stacked_features = src.to (device), mfe_targets.to (device), stacked_kmer_features.to (device), stacked_features.to (device)  
  
            targets = src  
            _, _, _, mfe_preds, vq_loss, denoising_loss = model (src, stacked_kmer_features, stacked_features, noise_std=noise_std)  
  
            mfe_pred = mfe_preds.squeeze ().cpu ().numpy ()  
            predicted_mfe.extend (mfe_pred)  
            true_mfe.extend (mfe_targets.cpu ().numpy ())  
  
            # 计算 MSE 损失  
            mse_loss = F.mse_loss (mfe_preds.squeeze (), mfe_targets.squeeze ()).item ()  
            mse_loss_sum += mse_loss * mfe_targets.size (0)  # 累加 MSE 损失和，同时考虑批次大小  
            num_samples += mfe_targets.size (0)  # 更新样本数量  
  
    # 计算平均 MSE 损失  
    #avg_mse_loss = mse_loss_sum /num_samples  
  
    #spearman_corr, _ = spearmanr (np.array (true_mfe), np.array (predicted_mfe))  
    #r2 = r2_score (true_mfe, predicted_mfe)  
    
   # 计算平均 MSE 损失
    avg_mse_loss = mse_loss_sum / num_samples
 
    # 计算 Spearman 相关系数
    spearman_corr, _ = spearmanr(np.array(true_mfe), np.array(predicted_mfe))
 
    # 计算 Pearson 相关系数
    pearson_corr, _ = pearsonr(np.array(true_mfe), np.array(predicted_mfe))
 
    # 计算 R^2 分数
    r2 = r2_score(true_mfe, predicted_mfe)
 
    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error(true_mfe, predicted_mfe))
 
    # 计算 MAE
    mae = mean_absolute_error(true_mfe, predicted_mfe)
 
    return spearman_corr, pearson_corr, r2, rmse, mae, avg_mse_loss


def calculate_sequence_lengths(file_name, vocab):  
    lengths = []  
    with open(file_name, 'r') as file:  
        for line in file:  
            if not line.startswith('>'):  
                rna_sequence = line.strip().upper()  
                lengths.append(len(rna_sequence))  
      
    return lengths  
class CustomLossBasedLRScheduler:
    def __init__(self, optimizer, initial_lr, loss_delta_threshold, lr_factor=0.5, patience=3):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.loss_delta_threshold = loss_delta_threshold
        self.lr_factor = lr_factor
        self.patience = patience
        self.counter = 0
        self.prev_loss = None
        self.current_lr = initial_lr
        self.reset()
 
    def reset(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr
        self.counter = 0
        self.prev_loss = None
 
    def step(self, current_loss):
        if self.prev_loss is None:
            self.prev_loss = current_loss
            return
 
        loss_delta = self.prev_loss - current_loss
        if loss_delta < self.loss_delta_threshold:
            self.counter += 1
            if self.counter >= self.patience:
                self.current_lr *= self.lr_factor
                self.counter = 0
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.current_lr
        else:
            self.counter = 0
 
        self.prev_loss = current_loss
class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model, trial_id):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, trial_id)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(val_loss, model, trial_id)

    def save_checkpoint(self, val_loss, model, trial_id):
        checkpoint_path = os.path.join(args.output_dir, f'best_model_trial_{trial_id}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        if self.verbose:
            print(f'Spearman improved ({self.best_score:.6f} --> {-val_loss:.6f}). Saving model ...')


vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3, 'A': 4, 'T': 5, 'G': 6, 'C': 7, '<CLS>': 8}
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for idx, char in enumerate(vocab)}
feature_cols = [
    'codon_H', 'codon_K', 'codon_I', 'uAUG', 'CGratio', 'codon_M', 'codon_E', 
    'codon_F', 'codon_D', 'codon_G', 'codon_T', 'codon_A', 'codon_C', 'codon_Q', 
    'codon_S', 'codon_L', 'codon_Y', 'codon_P', 'codon_R', 'codon_*','codon_V', 'codon_W', 
    'codon_N', 'uORF', 'CGperc', 'utrlen_m80', 'ATratio'  
]

def predict_mrl_and_compute_mfe(model, sequence, device):
    model.eval()
    with torch.no_grad():
        # Prepare input
        rna_seq_list = rna_to_idx(sequence)
        rna_seq_with_special = add_special_tokens(rna_seq_list, char_to_idx)
        input_tensor = torch.tensor([rna_seq_with_special], dtype=torch.long).to(device)
        
        # Compute features
        feature_dict = compute_features(sequence)
        features = [feature_dict[col] for col in feature_cols if col in feature_dict]
        # Pad or truncate features to 27
        while len(features) < 27:
            features.append(0.0)
        stacked_features = torch.tensor([features[:27]], dtype=torch.float).to(device)
        
        # K-mer features
        kmer_features = generate_kmer_features(sequence, k=3, num_features=128).unsqueeze(0).to(device)
        
        # Forward pass (noise_std=0 for prediction)
        _, _, _, mfe_pred, _, _ = model(input_tensor, kmer_features, stacked_features, noise_std=0.0)
        
        mrl_pred = mfe_pred.squeeze().item()  # Model predicts MRL (misnamed as mfe_pred)
        
        # Compute true MFE using ViennaRNA
        try:
            mfe, _ = compute_utr_mfe(sequence)
        except Exception as e:
            mfe = float('inf')  # Error handling
        
        # Additional features
        uaug = feature_dict.get('uAUG', 0)
        uorf = feature_dict.get('uORF', 0)
        cgperc = feature_dict.get('CGperc', 0)
        length = len(sequence)
        atratio = feature_dict.get('ATratio', 0)
        cgratio = feature_dict.get('CGratio', 0)
        
        return {
            'sequence': sequence,
            'mrl_pred': mrl_pred,
            'mfe': mfe,
            'uaug': uaug,
            'uorf': uorf,
            'cgperc': cgperc,
            'length': length,
            'atratio': atratio,
            'cgratio': cgratio
        }

def compute_mfe_wrapper(args):
    sequence = args
    try:
        mfe, _ = compute_utr_mfe(sequence)
    except Exception:
        mfe = float('inf')
    return mfe

def score_sequence(data):
    """
    为 5' UTR 序列计算综合评分，基于 MRL、MFE 和其他特征。
    
    参数:
        data (dict): 包含以下键的字典：
            - mrl_pred (float): 预测的 MRL 值。
            - mfe (float): 计算的 MFE 值（单位：kcal/mol）。
            - uaug (int): 上游 AUG 数量。
            - uorf (int): 上游 ORF 数量。
            - cgperc (float): GC 含量比例（0 到 1）。
            - length (int): 序列长度。
            - atratio (float): A/T 比率偏离 1 的绝对值。
            - cgratio (float): C/G 比率偏离 1 的绝对值。
    
    返回:
        float: 综合评分，值越高表示序列越优。
    """
    mrl_score = data['mrl_pred'] / 8.0  # 假设最大 MRL 为 8，归一化到 0-1
    mfe_score = (data['mfe'] + 100) / 100 if data['mfe'] != float('inf') else 0  # 归一化 MFE
    penalty = (data['uaug'] + data['uorf']) * 0.1  # 惩罚 uAUG 和 uORF
    cg_bonus = 1 if 0.4 <= data['cgperc'] <= 0.6 else 0.8  # 奖励适宜的 GC 含量
    length_penalty = 1 if 20 <= data['length'] <= 100 else 0.5  # 惩罚不适宜长度
    atratio_penalty = abs(data['atratio'] - 1) * 0.05  # 惩罚 A/T 比率不平衡
    cgratio_penalty = abs(data['cgratio'] - 1) * 0.05  # 惩罚 C/G 比率不平衡
    
    score = (0.6 * mrl_score + 0.3 * mfe_score - penalty - atratio_penalty - cgratio_penalty) * cg_bonus * length_penalty
    return score


# Function to process input sequence
def process_input_sequence(sequence, max_len=64):
    sequence = sequence.upper()
    seq_list = [char_to_idx.get(char, char_to_idx['<UNK>']) for char in sequence]
    seq_with_special = [char_to_idx['<CLS>']] + seq_list  # Add <CLS>
    if len(seq_with_special) > max_len:
        seq_with_special = seq_with_special[:max_len]
    else:
        seq_with_special += [char_to_idx['<EOS>']] * (max_len - len(seq_with_special))
    input_tensor = torch.tensor([seq_with_special], dtype=torch.long)  # Batch size 1
    return input_tensor

# Function to generate new sequence from model
def generate_new_sequence(model, input_seq, device, num_samples=1, temperature=1.0):
    model.eval()
    with torch.no_grad():
        input_tensor = process_input_sequence(input_seq).to(device)
        
        # Dummy kmer_features and features (since generation might not need them, or set to zeros)
        # Assuming kmer_feature_dim=128, and features len=27 from code
        kmer_features = torch.zeros(1, len(input_seq), 128).to(device)  # Dummy
        features = torch.zeros(1, 27).to(device)  # Dummy, 27 from feature_cols
        
        noise_std = 0.0  # No noise for inference
        
        logits, _, _, _, _, _ = model(input_tensor, kmer_features, features, noise_std=noise_std)
        
        # Sample from logits to generate new sequences
        generated_sequences = []
        for _ in range(num_samples):
            probs = F.softmax(logits / temperature, dim=-1)
            sampled_indices = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, -1)
            
            # Force prefix and suffix to guarantee start and end
            sampled_indices[0, 0] = char_to_idx['<CLS>']
            prefix = random.choice(['AGGA', 'AGGT'])
            prefix_ids = torch.tensor([char_to_idx[c] for c in prefix])
            sampled_indices[0, 1:5] = prefix_ids
            suffix = 'GCCACC'
            suffix_ids = torch.tensor([char_to_idx[c] for c in suffix])
            sampled_indices[0, 58:64] = suffix_ids
            
            generated_seq = ''.join([idx_to_char.get(idx.item(), '<UNK>') for idx in sampled_indices[0]])
            # Remove special tokens if needed
            generated_seq = generated_seq.replace('<PAD>', '').replace('<EOS>', '').replace('<CLS>', '').replace('<SOS>', '').replace('<UNK>', '')
            generated_sequences.append(generated_seq)
    
    return generated_sequences

# Batch prediction for MRL
def batch_predict_mrl(model, sequences, device, batch_size=32):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Batch predicting MRL"):
            batch_seqs = sequences[i:i+batch_size]
            batch_inputs = []
            batch_kmer = []
            batch_features = []
            for seq in batch_seqs:
                rna_seq_list = rna_to_idx(seq)
                rna_seq_with_special = add_special_tokens(rna_seq_list, char_to_idx)
                input_tensor = torch.tensor(rna_seq_with_special, dtype=torch.long)
                batch_inputs.append(input_tensor)
                
                feature_dict = compute_features(seq)
                features = [feature_dict[col] for col in feature_cols if col in feature_dict]
                while len(features) < 27:
                    features.append(0.0)
                batch_features.append(torch.tensor(features[:27], dtype=torch.float))
                
                kmer_feat = generate_kmer_features(seq, k=3, num_features=128)
                batch_kmer.append(kmer_feat)
            
            # Pad to max length in batch
            max_len = max(len(s) for s in batch_inputs)
            padded_inputs = torch.full((len(batch_inputs), max_len), char_to_idx['<PAD>'], dtype=torch.long)
            for j, inp in enumerate(batch_inputs):
                padded_inputs[j, :len(inp)] = inp
            
            padded_kmer = torch.nn.utils.rnn.pad_sequence(batch_kmer, batch_first=True, padding_value=0.0)
            padded_features = torch.stack(batch_features)
            
            padded_inputs = padded_inputs.to(device)
            padded_kmer = padded_kmer.to(device)
            padded_features = padded_features.to(device)
            
            _, _, _, mfe_preds, _, _ = model(padded_inputs, padded_kmer, padded_features, noise_std=0.0)
            
            mrl_preds = mfe_preds.squeeze().cpu().numpy()
            all_predictions.extend(mrl_preds)
    
    return all_predictions

# Main generation and screening code

def resolve_checkpoint_state_dict(checkpoint):
    """Return a pure model state_dict from raw or training-dict checkpoints."""
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model", "module"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
        else:
            if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
                state_dict = checkpoint
            else:
                raise KeyError(
                    "Checkpoint does not look like a pure model state_dict and no known weight key "
                    "was found. Expected one of: model_state_dict, state_dict, model, module."
                )
    else:
        state_dict = checkpoint

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def infer_5utr_hparams_from_state_dict(state_dict, args):
    """Infer architecture-critical hyperparameters when possible."""
    inferred = {}
    if "embedding.weight" in state_dict:
        inferred["embed_dim"] = int(state_dict["embedding.weight"].shape[1])
    if "fc1.weight" in state_dict:
        inferred["hidden_dim"] = int(state_dict["fc1.weight"].shape[0])
    if "fc21.weight" in state_dict:
        inferred["latent_dim"] = int(state_dict["fc21.weight"].shape[0])
    if "vector_quantizer.weight" in state_dict:
        inferred["num_embeddings"] = int(state_dict["vector_quantizer.weight"].shape[0])
        inferred["latent_dim"] = int(state_dict["vector_quantizer.weight"].shape[1])

    changed = []
    for k, v in inferred.items():
        if hasattr(args, k) and getattr(args, k) != v:
            changed.append((k, getattr(args, k), v))
            setattr(args, k, v)

    if changed:
        print("Auto-matched model hyperparameters from checkpoint:")
        for k, old, new in changed:
            print(f"  {k}: {old} -> {new}")
    return args


def clean_5utr_sequence(seq):
    seq = str(seq).strip().upper().replace("U", "T")
    seq = re.sub(r"[^ATGCN]", "", seq)
    return seq


def read_first_fasta_sequence(path):
    records = list(SeqIO.parse(path, "fasta"))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return str(records[0].seq)


def minmax_normalize(values, higher_is_better=True):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=float)
    if not finite.any():
        return out
    v = arr[finite]
    vmin, vmax = float(v.min()), float(v.max())
    if abs(vmax - vmin) < 1e-12:
        out[finite] = 0.5
    else:
        out[finite] = (v - vmin) / (vmax - vmin)
    if not higher_is_better:
        out[finite] = 1.0 - out[finite]
    return out


def gc_window_score(gc_values, low=0.40, high=0.60):
    scores = []
    center = (low + high) / 2.0
    half_width = (high - low) / 2.0
    for gc in gc_values:
        if low <= gc <= high:
            scores.append(1.0)
        else:
            distance = abs(gc - center) - half_width
            scores.append(max(0.0, 1.0 - distance / 0.50))
    return np.asarray(scores, dtype=float)


def build_5utr_score_table(sequences, mrl_preds, mfe_values,
                           mrl_weight=0.60, mfe_weight=0.15, gc_weight=0.10,
                           penalty_weight=0.15, mfe_direction="higher"):
    rows = []
    for seq, mrl_pred, mfe in zip(sequences, mrl_preds, mfe_values):
        feat = compute_features(seq)
        rows.append({
            "sequence": seq,
            "length": len(seq),
            "mrl_pred": float(mrl_pred),
            "mfe": float(mfe) if np.isfinite(mfe) else np.nan,
            "uAUG": int(feat.get("uAUG", 0)),
            "uORF": int(feat.get("uORF", 0)),
            "GC": float(feat.get("CGperc", 0.0)),
            "CGratio": float(feat.get("CGratio", 0.0)),
            "ATratio": float(feat.get("ATratio", 0.0)),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["mrl_score"] = minmax_normalize(df["mrl_pred"].values, higher_is_better=True)

    # For 5' UTR expression/MRL, very stable structures can suppress scanning/initiation.
    # Default "higher" means less negative MFE is better. Use lower for stability-oriented screening.
    df["mfe_score"] = minmax_normalize(df["mfe"].fillna(df["mfe"].min()).values,
                                       higher_is_better=(mfe_direction == "higher"))
    df["gc_score"] = gc_window_score(df["GC"].values)
    df["uAUG_uORF_penalty"] = penalty_weight * (df["uAUG"] + df["uORF"])

    df["final_score"] = (
        mrl_weight * df["mrl_score"]
        + mfe_weight * df["mfe_score"]
        + gc_weight * df["gc_score"]
        - df["uAUG_uORF_penalty"]
    )
    return df.sort_values("final_score", ascending=False).reset_index(drop=True)


def save_fasta(df, path, sequence_col="sequence", prefix="candidate"):
    with open(path, "w") as f:
        for i, row in df.iterrows():
            f.write(f">{prefix}_{i+1}|final_score={row['final_score']:.6f}|mrl_pred={row['mrl_pred']:.6f}|mfe={row['mfe']:.6f}\n")
            f.write(str(row[sequence_col]) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone 5' UTR generation, prediction, MFE/feature scoring, and Top-k selection"
    )
    parser.add_argument("--checkpoint", type=str, default=os.path.join(os.path.dirname(__file__), "Model.pth"),
                        help="One checkpoint used for both generation and MRL prediction")
    parser.add_argument("--input_seq", type=str, default=None,
                        help="Input/template 5' UTR sequence. Use this for a raw sequence string.")
    parser.add_argument("--input_fasta", type=str, default=None,
                        help="Input FASTA file. The first record is used as template.")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of candidate 5' UTRs to generate")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top candidates to output")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_embeddings", type=int, default=32)
    parser.add_argument("--commitment_cost", type=float, default=0.1)
    parser.add_argument("--min_len", type=int, default=20)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--prefixes", type=str, default="AGGA,AGGT",
                        help="Comma-separated allowed prefixes. Use empty string to disable prefix filtering.")
    parser.add_argument("--suffix", type=str, default="GCCACC",
                        help="Required suffix. Use empty string to disable suffix filtering.")
    parser.add_argument("--mrl_weight", type=float, default=0.60)
    parser.add_argument("--mfe_weight", type=float, default=0.15)
    parser.add_argument("--gc_weight", type=float, default=0.10)
    parser.add_argument("--penalty_weight", type=float, default=0.15,
                        help="Penalty multiplier for uAUG+uORF")
    parser.add_argument("--mfe_direction", type=str, default="higher", choices=["higher", "lower"],
                        help="For 5'UTR MRL/expression, higher/less negative MFE is usually safer; use lower for stability screening.")
    parser.add_argument("--output_dir", type=str, default="./generated_5utr_top10")
    parser.add_argument("--output_prefix", type=str, default="generated_5utr")

    args = parser.parse_args()

    if args.input_seq:
        input_seq = clean_5utr_sequence(args.input_seq)
    elif args.input_fasta:
        input_seq = clean_5utr_sequence(read_first_fasta_sequence(args.input_fasta))
    else:
        input_seq = clean_5utr_sequence("AGGAATAAACTAGTATTCTTCTGGTCCCCACAGACTCAGAGAGAACCCGCCACC")

    if not input_seq:
        raise ValueError("Input 5' UTR sequence is empty after cleaning.")

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = resolve_checkpoint_state_dict(checkpoint)
    args = infer_5utr_hparams_from_state_dict(state_dict, args)

    model = VAEWithTransformer(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        kmer_feature_dim=128,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Input 5' UTR: {input_seq}")
    print(f"Generating {args.num_samples} candidates...")

    generated = generate_new_sequence(
        model,
        input_seq,
        device,
        num_samples=args.num_samples,
        temperature=args.temperature,
    )

    generated = [clean_5utr_sequence(s) for s in generated]
    generated = [s for s in generated if s and "<UNK>" not in s]

    prefixes = [p.strip().upper() for p in args.prefixes.split(",") if p.strip()]
    valid = []
    for s in generated:
        if not (args.min_len <= len(s) <= args.max_len):
            continue
        if prefixes and not any(s.startswith(p) for p in prefixes):
            continue
        if args.suffix and not s.endswith(args.suffix.upper()):
            continue
        valid.append(s)

    # Preserve order while removing exact duplicates.
    valid = list(dict.fromkeys(valid))

    if len(valid) == 0:
        print("Warning: no candidates passed filtering. Falling back to unfiltered generated candidates.")
        valid = list(dict.fromkeys(generated))

    print(f"Valid candidates: {len(valid)}")

    mrl_preds = batch_predict_mrl(model, valid, device, batch_size=args.batch_size)

    if len(valid) > 1:
        num_processes = max(1, min(cpu_count(), len(valid)))
        with Pool(processes=num_processes) as pool:
            mfe_values = list(tqdm(pool.imap(compute_mfe_wrapper, valid), total=len(valid), desc="Computing MFE"))
    else:
        mfe_values = [compute_mfe_wrapper(valid[0])] if valid else []

    scored = build_5utr_score_table(
        valid,
        mrl_preds,
        mfe_values,
        mrl_weight=args.mrl_weight,
        mfe_weight=args.mfe_weight,
        gc_weight=args.gc_weight,
        penalty_weight=args.penalty_weight,
        mfe_direction=args.mfe_direction,
    )

    if scored.empty:
        raise RuntimeError("No generated sequence is available for scoring.")

    top = scored.head(args.top_k).copy()

    all_csv = os.path.join(args.output_dir, f"{args.output_prefix}_all_scored.csv")
    top_csv = os.path.join(args.output_dir, f"{args.output_prefix}_top{args.top_k}.csv")
    top_fasta = os.path.join(args.output_dir, f"{args.output_prefix}_top{args.top_k}.fasta")

    scored.to_csv(all_csv, index=False)
    top.to_csv(top_csv, index=False)
    save_fasta(top, top_fasta, prefix="top_5utr")

    print(f"Saved all scored candidates: {all_csv}")
    print(f"Saved Top{args.top_k} CSV: {top_csv}")
    print(f"Saved Top{args.top_k} FASTA: {top_fasta}")
    print("\nTop candidates:")
    for i, row in top.iterrows():
        print(
            f"{i+1}: {row['sequence']} | MRL={row['mrl_pred']:.4f} | "
            f"MFE={row['mfe']:.4f} | GC={row['GC']:.4f} | "
            f"uAUG={int(row['uAUG'])} | uORF={int(row['uORF'])} | "
            f"final_score={row['final_score']:.4f}"
        )


if __name__ == "__main__":
    main()
