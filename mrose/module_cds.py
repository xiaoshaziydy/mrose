import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from collections import defaultdict
import random
from torch.utils.checkpoint import checkpoint_sequential

# ==========================================
# 1. Basic data structures and helper functions
# ==========================================

DEFAULT_CODON_TABLE = {
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

def rna_to_idx(rna_seq, codon_table, char_to_idx):
    codons = [rna_seq[i:i+3] for i in range(0, len(rna_seq) - 2, 3)]
    return [char_to_idx.get(codon, char_to_idx.get('<UNK>', 0)) for codon in codons]

def idx_to_rna(idx_seq, idx_to_char):
    return ''.join([idx_to_char.get(idx, '<UNK>') for idx in idx_seq])

def add_special_tokens(rna_seq_list, char_to_idx):
    return [char_to_idx['<SOS>']] + rna_seq_list + [char_to_idx['<EOS>']]

def pad_or_truncate(seq, max_len, char_to_idx):
    if len(seq) > max_len:
        return seq[:max_len]
    else:
        return seq + [char_to_idx['<PAD>']] * (max_len - len(seq))

def nuc_to_idx(nuc_seq, nuc_char_to_idx):
    return [nuc_char_to_idx.get(nuc, nuc_char_to_idx.get('N', 0)) for nuc in nuc_seq]

def add_special_tokens_nuc(nuc_seq_list, nuc_char_to_idx):
    return [nuc_char_to_idx['<SOS>']] + nuc_seq_list + [nuc_char_to_idx['<EOS>']]

def pad_or_truncate_nuc(seq, max_len, nuc_char_to_idx):
    if len(seq) > max_len:
        return seq[:max_len]
    else:
        return seq + [nuc_char_to_idx['<PAD>']] * (max_len - len(seq))

# ==========================================
# 2. Deep-learning core components
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
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=padding1) 
        self.bn1 = NoiseAwareBatchNorm1d(hidden_dim)
        self.se1 = SEBlock(hidden_dim)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.transformer = TransformerBlock(hidden_dim, hidden_dim, num_heads, num_layers, hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, out_channels, kernel_size=kernel_size, padding=padding2, dilation=1) 
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
    def __init__(self, in_channels, embed_dim, n_blocks=4, base_channels=16, num_heads=2, num_layers=2, hidden_dim=128, max_len=256):
        super(ComplexTransformerUNet1D, self).__init__()
        self.target_len = max_len // 3
        out_channels = embed_dim
        self.inc = DoubleConv1DWithTransformer(in_channels, base_channels * 8, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)

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
        self.bottleneck = BottleneckAttention(base_channels * 32, num_heads=2)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='linear'),
            UpWithTransformer(base_channels * 32, base_channels * 16, skip_channels=base_channels * 16, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='linear'),
            UpWithTransformer(base_channels * 16, base_channels * 8, skip_channels=base_channels * 16, num_heads=num_heads, num_layers=num_layers, hidden_dim=hidden_dim)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='linear'),
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
        
        x = self.bottleneck(x6)
        
        x = self.up1[0](x)
        x = self.up1[1](x, x5, noise_std)
        x = self.up2[0](x)
        x = self.up2[1](x, x4,noise_std)
        x = self.up3[0](x)
        x = self.up3[1](x, x1, noise_std)

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
            raise ValueError(f"Sequence length {x.size(1)} exceeds max_len {self.pe.size(1)}")
        x = x + self.pe[:, :x.size(1), :]
        return x

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
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
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
# 3. Core CDS model module renamed for integration with the main program
# ==========================================

class Model_CDS(nn.Module):
    """
    Fully reconstructed based on VAEWithTransformer.
    For compatibility with the single-input main_fusion path, which only provides codon-level src, if src_nuc is not provided,
    the model automatically creates a safe padded tensor for the dual-stream architecture and outputs deeply fused cnn_features latent variables.
    """
    def __init__(self, vocab_size=68, embed_dim=32, hidden_dim=32, latent_dim=32, 
                 nuc_vocab_size=8, num_encoder_layers=2, num_decoder_layers=2, 
                 num_heads=8, num_embeddings=32, commitment_cost=0.01, 
                 kmer_feature_dim=128, max_len=256, codon_table=None, 
                 char_to_idx=None, nuc_char_to_idx=None):
        
        super(Model_CDS, self).__init__()
        self.vocab_size = vocab_size
        self.nuc_vocab_size = nuc_vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_len = max_len
        
        # Compatibility default table
        self.codon_table = codon_table if codon_table else DEFAULT_CODON_TABLE
        self.char_to_idx = char_to_idx if char_to_idx else {'<PAD>': 0, '<UNK>': 3}
        self.nuc_char_to_idx = nuc_char_to_idx if nuc_char_to_idx else {'<PAD>': 0, 'N': 7}
        
        self.vector_quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.nuc_embedding = nn.Embedding(nuc_vocab_size, embed_dim)
        self.embed_scale = nn.Parameter(torch.tensor(math.sqrt(embed_dim)))
        
        # Codon-aware positional encoding
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
        
        encoder_layer = EnhancedEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, dropout=0.2, max_relative_positions=8)
        self.transformer_encoder = EnhancedTransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.nuc_transformer_encoder = EnhancedTransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.nuc_encoder_norm = nn.LayerNorm(embed_dim)
        
        self.fc1 = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01)
        )
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, embed_dim)
        
        self.decoder_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_decoder = CodonAwarePositionalEncoding(embed_dim, max_len=self.max_len // 3)
        
        decoder_layer = EnhancedTransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=256, max_relative_positions=8)
        self.transformer_decoder = EnhancedTransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
        self.nuc_decoder_embedding = nn.Embedding(nuc_vocab_size, embed_dim)
        self.nuc_pos_decoder = PositionalEncoding(embed_dim, max_len=self.max_len)
        self.nuc_transformer_decoder = EnhancedTransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_nuc_out = nn.Linear(embed_dim, nuc_vocab_size)
        
        self.cnn_feature_extractor0 = CNNFeatureExtractor(embed_dim, num_filters=latent_dim, filter_size=3)
        self.cnn_feature_extractor = CNNFeatureExtractor(embed_dim*4, num_filters=latent_dim, filter_size=3)
        
        self.unet = ComplexTransformerUNet1D(embed_dim, embed_dim, base_channels=64, hidden_dim=128, max_len=self.max_len)
        self.unet1 = ComplexTransformerUNet1D(embed_dim, embed_dim, base_channels=64, hidden_dim=128, max_len=self.max_len*3)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.nuc_downsample_conv = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=3, padding=0)
        nn.init.kaiming_normal_(self.nuc_downsample_conv.weight, mode='fan_in', nonlinearity='relu')

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
            padding = torch.zeros(nuc_encoder_output.size(0), target_len - nuc_encoder_output.size(1), nuc_encoder_output.size(2), device=nuc_encoder_output.device)
            nuc_encoder_output = torch.cat([nuc_encoder_output, padding], dim=1)
        return nuc_encoder_output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, src, max_len=256):
        batch_size, seq_len = src.size()
        target_len = max_len // 3
        if seq_len > target_len:
            src = src[:, :target_len]
        else:
            pad_size = target_len - seq_len
            src = F.pad(src, (0, pad_size), value=self.char_to_idx.get('<PAD>', 0))
        
        seq_len = target_len
        src_mask = self._generate_square_subsequent_mask(seq_len).to(src.device)
        src_key_padding_mask = (src == self.char_to_idx.get('<PAD>', 0))
        
        src_emb = self.embedding(src) * self.embed_scale
        src_emb = self.norm1(src_emb)
        src_emb = self.pos_encoder(src_emb)
        
        src_conv = src_emb.permute(0, 2, 1)
        conv_outputs = [conv(src_conv) for conv in self.conv_layers]
        conv_outputs = [o.permute(0, 2, 1) for o in conv_outputs]
        src_conv = sum(conv_outputs)
        
        fused_emb = src_emb + src_conv
        encoder_input = self.encoder_norm(fused_emb)
        encoder_output = self.transformer_encoder(encoder_input, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
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
            src_nuc = F.pad(src_nuc, (0, pad_size), value=self.nuc_char_to_idx.get('<PAD>', 0))
        
        seq_len = target_len
        src_mask = self._generate_square_subsequent_mask(seq_len).to(src_nuc.device)
        src_key_padding_mask = (src_nuc == self.nuc_char_to_idx.get('<PAD>', 0))
        
        src_emb = self.nuc_embedding(src_nuc) * self.embed_scale
        src_emb = self.norm1(src_emb)
        src_emb = self.nuc_pos_encoder(src_emb)
        
        src_conv = src_emb.permute(0, 2, 1)
        conv_outputs = [conv(src_conv) for conv in self.nuc_conv_layers]
        conv_outputs = [o.permute(0, 2, 1) for o in conv_outputs]
        src_conv = sum(conv_outputs)
        
        fused_emb = src_emb + src_conv
        encoder_input = self.nuc_encoder_norm(fused_emb)
        encoder_output = self.nuc_transformer_encoder(encoder_input, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        encoder_output = encoder_input + encoder_output
        encoder_output = self.nuc_encoder_norm(encoder_output)
        return encoder_output

    def forward(self, src, src_nuc=None, noise_std=0.0):
        if src_nuc is None:
            src_nuc = torch.full((src.size(0), src.size(1) * 3), self.nuc_char_to_idx.get('<PAD>', 0), dtype=torch.long, device=src.device)

        mu, log_var, encoder_output = self.encode(src, max_len=self.max_len)
        nuc_encoder_output = self.encode_nuc(src_nuc, max_len=self.max_len)

        denoised_output_gauss = self.unet(encoder_output.permute(0, 2, 1), noise_std).permute(0, 2, 1)
        denoised_output_bio = self.unet(encoder_output.permute(0, 2, 1), noise_std).permute(0, 2, 1)

        denoised_nuc_output_gauss = self.unet1(nuc_encoder_output.permute(0, 2, 1), noise_std).permute(0, 2, 1)
        denoised_nuc_output_bio = self.unet1(nuc_encoder_output.permute(0, 2, 1), noise_std).permute(0, 2, 1)

        denoised_nuc_output_gauss_down = self.downsample_nuc_encoder_output(denoised_nuc_output_gauss, self.max_len)
        denoised_nuc_output_bio_down = self.downsample_nuc_encoder_output(denoised_nuc_output_bio, self.max_len)

        # ================= NEW: Extract tensors required for CDS loss computation =================
        denoising_loss = F.mse_loss(denoised_output_gauss, encoder_output) + F.mse_loss(denoised_nuc_output_gauss, nuc_encoder_output)
        
        codon_rep = encoder_output.mean(dim=1)
        nuc_rep = nuc_encoder_output.mean(dim=1)

        # 1. Codon decoding (for reconstruction)
        tgt_seq_len = src.size(1) if src.size(1) <= self.max_len // 3 else self.max_len // 3
        emb = self.decoder_embedding(torch.zeros((src.size(0), tgt_seq_len), dtype=torch.long, device=src.device))
        emb = emb * math.sqrt(self.embed_dim)
        emb = self.pos_decoder(emb)
        decoder_out = self.transformer_decoder(emb, denoised_output_gauss)
        logits = self.fc_out(decoder_out)

        # 2. Nucleotide decoding (for reconstruction)
        tgt_nuc_len = src_nuc.size(1) if src_nuc.size(1) <= self.max_len else self.max_len
        nuc_emb = self.nuc_decoder_embedding(torch.zeros((src_nuc.size(0), tgt_nuc_len), dtype=torch.long, device=src.device))
        nuc_emb = nuc_emb * math.sqrt(self.embed_dim)
        nuc_emb = self.nuc_pos_decoder(nuc_emb)
        nuc_decoder_out = self.nuc_transformer_decoder(nuc_emb, denoised_nuc_output_gauss)
        nuc_logits = self.fc_nuc_out(nuc_decoder_out)
        # =================================================================

        combined_features = torch.cat([
            denoised_output_gauss.permute(0, 2, 1),
            denoised_output_bio.permute(0, 2, 1),
            denoised_nuc_output_gauss_down.permute(0, 2, 1),
            denoised_nuc_output_bio_down.permute(0, 2, 1),
        ], dim=1)
        
        cnn_features = self.cnn_feature_extractor(combined_features)
        
        # Return the main features and all tensors required for loss computation
        return cnn_features, logits, nuc_logits, mu, log_var, denoising_loss, codon_rep, nuc_rep

