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

class MLP(nn.Module):
    """
    Multi-layer perceptron with latent space pruning. 
    """
    def __init__(self, d_embed, expansion_factor=4, dropout=0.1):
        super().__init__()

        d_latent = d_embed * expansion_factor

        self.W_up = nn.Parameter(torch.empty(d_embed, d_latent))
        self.W_down = nn.Parameter(torch.empty(d_latent, d_embed))

        self.bias_up = nn.Parameter(torch.empty(d_latent))
        self.bias_down = nn.Parameter(torch.empty(d_embed))

        self.latent_mask = torch.ones(d_latent, dtype=torch.bool)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        
        # Initialize weights

        nn.init.kaiming_uniform_(self.W_up, a=sqrt(5))
        nn.init.kaiming_uniform_(self.W_down, a=sqrt(5))

        # Initialize biases

        fan_in_up, _ = nn.init._calculate_fan_in_and_fan_out(self.W_up)
        bound_up = 1 / sqrt(fan_in_up) if fan_in_up > 0 else 0
        nn.init.uniform_(self.bias_up, -bound_up, bound_up)

        fan_in_down, _ = nn.init._calculate_fan_in_and_fan_out(self.W_down)
        bound_down = 1 / sqrt(fan_in_down) if fan_in_down > 0 else 0
        nn.init.uniform_(self.bias_down, -bound_down, bound_down)

    def pruned_mlp_matrices(self):
        """
        Returns pruned MLP matrices
        """

        W_up = self.W_up[:, self.latent_mask]
        bias_up = self.bias_up[self.latent_mask]
        W_down = self.W_down[self.latent_mask, :]

        return W_up, bias_up, W_down, self.bias_down

    def forward(self, e):
        """
        Input:
            e: Embeddings of shape (n_batch, n_tokens, d_embed)
        
        Output:
            e: Updated embeddings of shape (n_batch, n_tokens, d_embed)
        """

        # Get pruned MLP matrices

        W_up, bias_up, W_down, bias_down = self.pruned_mlp_matrices()

        e = e @ W_up + bias_up
        e = self.activation(e)
        e = self.dropout(e)
        e = e @ W_down + bias_down
        e = self.dropout(e)

        return e

class TransformerBlock(nn.Module):
    """
    Transformer block with pre-block layer normalization. 
    """
    def __init__(self, d_embed: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(d_embed)
        self.attn = MHSDPAttention(d_embed, n_heads)

        self.norm2 = nn.LayerNorm(d_embed)
        self.mlp = MLP(d_embed, expansion_factor=4, dropout=dropout)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """
        Input:
            e: Embeddings of shape (n_batch, n_tokens, d_embed)
        
        Output:
            e: Updated embeddings of shape (n_batch, n_tokens, d_embed)
        """
        e = e + self.attn(self.norm1(e))
        e = e + self.mlp(self.norm2(e))
        return e
