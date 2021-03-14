import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def make_tta(x, x_len, dtype):
    tta = torch.zeros(x.size(0), x.size(1), device=x.device, dtype=dtype)
    for i in range(0, x.size(0)):
        if x_len[i] >= x.size(1):
            tta[i, :] = torch.arange(x_len[i], x_len[i] - x.size(1), -1, device=x.device)
        else:
            tta[i, :x_len[i]] = torch.arange(x_len[i], 0, -1, device=x.device)
    return tta


class FcEncoder(nn.Module):
    def __init__(self, indim, hdim, layers, dropout=0.1):
        super(FcEncoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(nn.Linear(indim, hdim),
            nn.ReLU(),
            nn.Dropout(dropout))
        )
        for _ in range(layers):
            self.encoder.append(nn.Sequential(
                nn.Linear(hdim, hdim),
                nn.ReLU(),
                nn.Dropout(dropout))
            )

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x


class Generator(nn.Module):
    def __init__(self, d_model, out_dim):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, x):
        return self.proj(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=121):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(math.log(10000.0) * torch.arange(0, d_model, 2) / d_model)
        pe[:, 0::2] = torch.sin(pos / div_term)
        pe[:, 1::2] = torch.cos(pos / div_term)
        pe[0] = 0.
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, x, x_len):
        ind = make_tta(x, x_len, dtype=torch.long)
        ind = torch.clamp(ind, max=self.max_len-1)
        x = x + self.pe[ind]
        return self.dropout(x)


class TargetNoise(nn.Module):
    def __init__(self, sigma, min=5, max=60):
        super(TargetNoise, self).__init__()
        self.sigma = sigma
        self.min = min
        self.max = max

    def forward(self, x, x_len):
        tta = make_tta(x, x_len, dtype=torch.float32)
        
        _mask_1 = torch.bitwise_and(tta >= self.min, tta < self.max)
        _mask_2 = (tta >= self.max)
        _mask_3 = (tta < self.min)
        tta[_mask_1] = (tta[_mask_1] - self.min) / (self.max - self.min)
        tta[_mask_2] = 1.
        tta[_mask_3] = 0.
        return torch.randn(x.size(), device=x.device) * self.sigma * tta.unsqueeze(-1)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, x_mask):
        for layer in self.layers:
            x = layer(x, x_mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, x_mask):
        x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, x_mask))
        return self.sublayer[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def attention(query, key, value, L, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # if mask is not None:
    #     scores = scores.masked_fill(mask == 0, -1e9)
    input_len = mask.sum(dim=-1).max(dim=-1)[0]
    M = torch.zeros_like(scores)
    maskn = L.shape[0]//2
    for batchn in range(scores.shape[0]):
        ks = input_len[batchn][0]
        for p in range(scores.shape[1]):
            for i in range(0,ks):
                for j in range(max(0,i-maskn),min(ks-1,i+maskn)+1):
                    M[batchn][p][i][j] = L[maskn-(i-j)]
    scores = scores * M
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, Ln, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.L = torch.tensor([np.exp(-(x) ** 2 / 2) / (math.sqrt(2 * math.pi) ) for x in range (-(Ln//2),Ln//2+1,1)])
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        x, self.attn = attention(query, key, value, self.L,  mask=mask,
                                 dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class TGModel(nn.Module):
    def __init__(self, cfg):
        super(TGModel, self).__init__()
        
        state_dim = cfg.state_dim
        offset_dim = cfg.offset_dim
        target_dim = cfg.target_dim
        out_dim = cfg.out_dim
        hdim = cfg.encoder.hidden_dim
        enc_layers = cfg.encoder.layers
        pe_max_len = cfg.max_trans + 1
        tgt_noise_sigma = cfg.tgt_noise.sigma
        tgt_noise_min = cfg.tgt_noise.min
        tgt_noise_max = cfg.tgt_noise.max
        attn_h = cfg.transformer.attn_h
        d_ff = cfg.transformer.d_ff
        tf_layers = cfg.transformer.layers
        Ln = cfg.transformer.Ln
        dropout = cfg.dropout

        self.state_encoder = FcEncoder(state_dim, hdim, enc_layers, dropout=dropout)
        self.offset_encoder = FcEncoder(offset_dim, hdim, enc_layers, dropout=dropout)
        self.target_encoder = FcEncoder(target_dim, hdim, enc_layers, dropout=dropout)

        self.tta_embedding = PositionalEncoding(hdim, dropout, max_len=pe_max_len)
        self.target_noise = TargetNoise(tgt_noise_sigma, tgt_noise_min, tgt_noise_max)

        hdim *= 3
        layer = EncoderLayer(hdim, MultiHeadedAttention(attn_h, hdim, Ln, dropout), PositionwiseFeedForward(hdim, d_ff, dropout), dropout)
        self.transformer = Encoder(layer, tf_layers)

        self.decoder = Generator(hdim, out_dim)

    def forward(self, state, offset, target, mask, input_len, add_tgt_noise=True):
        state = self.state_encoder(state)
        offset = self.offset_encoder(offset)
        target = self.target_encoder(target)
        state = self.tta_embedding(state, input_len)
        offset = self.tta_embedding(offset, input_len)
        target = self.tta_embedding(target, input_len)

        h = torch.cat((offset, target), dim=-1)
        if add_tgt_noise:
            h = h + self.target_noise(h, input_len)
        h = torch.cat((state, h), dim=-1)

        h = self.transformer(h, mask)

        return self.decoder(h)
