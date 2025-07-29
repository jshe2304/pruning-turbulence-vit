import torch
import torch.nn as nn

from .attention import Attention
from .mlp import MLP

class TransformerBlock(nn.Module):
    """
    Transformer block with pre-block layer normalization. 
    """
    def __init__(self, d_embed: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(d_embed)
        self.attn = Attention(d_embed, n_heads)

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
