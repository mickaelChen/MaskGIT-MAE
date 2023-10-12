# BERT architecture for the Masked Bidirectional Encoder Transformer
import torch
from torch import nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super(Attention, self).__init__()
        self.dim = embed_dim
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True, bias=True)

    def forward(self, x):
        attention_value, attention_weight = self.mha(x, x, x)
        return attention_value, attention_weight


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        l_attn = []
        for attn, ff in self.layers:
            attention_value, attention_weight = attn(x)
            x = attention_value + x
            x = ff(x) + x
            l_attn.append(attention_weight)
        return x, l_attn


class MaskTransformer(nn.Module):
    def __init__(self, img_size=256, hidden_dim=768, codebook_size=1024, depth=24, heads=8, mlp_dim=3072, dropout=0.1, nclass=1000):
        super().__init__()

        self.nclass = nclass
        self.patch_size = img_size // 16
        self.codebook_size = codebook_size
        self.tok_emb = nn.Embedding(codebook_size+1+nclass+1, hidden_dim)  # +1 for the mask of the viz token, +1 for mask of the class
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, (self.patch_size*self.patch_size)+1, hidden_dim)), 0., 0.02)
        self.first_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
        )

        self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        self.last_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
        )

        self.bias = nn.Parameter(torch.zeros((self.patch_size*self.patch_size)+1, codebook_size+1+nclass+1))

    def forward(self, img_token, y=None, drop_label=None, return_attn=False):
        b, w, h = img_token.size()

        cls_token = y.view(b, -1) + self.codebook_size + 1
        cls_token[drop_label] = self.codebook_size + 1 + self.nclass
        input = torch.cat([img_token.view(b, -1), cls_token.view(b, -1)], -1)
        tok_embeddings = self.tok_emb(input)
        pos_embeddings = self.pos_emb

        x = tok_embeddings + pos_embeddings

        x = self.first_layer(x)
        x, attn = self.transformer(x)
        x = self.last_layer(x)

        logit = torch.matmul(x, self.tok_emb.weight.T) + self.bias

        if return_attn:
            return logit[:, :self.patch_size * self.patch_size, :self.codebook_size + 1], attn

        return logit[:, :self.patch_size*self.patch_size, :self.codebook_size+1]
