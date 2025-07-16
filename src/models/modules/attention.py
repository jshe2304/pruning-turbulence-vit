import torch
import torch.nn as nn
from math import sqrt

class MHSDPAttention(nn.Module):
    """
    Multi-head scaled dot-product attention with attention head pruning. 
    """
    def __init__(self, d_embed: int, n_heads: int) -> None:
        super().__init__()

        assert d_embed % n_heads == 0

        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_attn = d_embed // n_heads
        self.scale = self.d_attn ** -0.5

        self.QKV = nn.Parameter(torch.empty(d_embed, d_embed * 3))
        self.out_proj = nn.Parameter(torch.empty(d_embed, d_embed))

        self.head_mask = torch.ones(n_heads, dtype=torch.bool)

        self.reset_parameters()

    @property
    def n_unpruned_heads(self) -> int:
        return int(torch.sum(self.head_mask).item())

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.QKV, a=sqrt(5))
        nn.init.kaiming_normal_(self.out_proj, a=sqrt(5))

    def pruned_attention_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns pruned QKV and output projection matrices
        """
        if torch.all(self.head_mask): return self.QKV, self.out_proj

        QKV = self.QKV.reshape(self.d_embed, self.n_heads, 3 * self.d_attn)
        QKV = QKV[:, self.head_mask]
        QKV = QKV.reshape(self.d_embed, 3 * self.n_unpruned_heads * self.d_attn)

        W = self.out_proj.reshape(self.n_heads, self.d_attn, self.d_embed)
        W = W[self.head_mask]
        W = W.reshape(self.n_unpruned_heads * self.d_attn, self.d_embed)

        return QKV, W

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """
        Input:
            e: Token embeddings
        
        Output:
            residual: Residual stream update
        """

        n_batch, n_tokens, d_embed = e.size()

        # Get pruned attention matrices

        QKV, W = self.pruned_attention_matrices()

        # Compute Q, K, V matrices efficiently

        qkv = e @ QKV
        qkv = qkv.reshape(n_batch, n_tokens, self.n_unpruned_heads, 3 * self.d_attn)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        # Compute attention pattern

        attn = q @ k.transpose(-2, -1) * self.scale
        attn = torch.softmax(attn, dim=-1)

        # Compute values

        values = attn @ v
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(n_batch, n_tokens, self.n_unpruned_heads * self.d_attn)
        
        # Map out of attention space

        residual = values @ W

        return residual
