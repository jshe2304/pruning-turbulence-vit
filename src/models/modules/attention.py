import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, d_embed, n_heads) -> None:

        super().__init__()
        assert d_embed % n_heads == 0

        self.n_heads = n_heads
        self.attn_dim = d_embed // n_heads
        self.scale = self.attn_dim ** -0.5

        self.qkv = nn.Linear(d_embed, d_embed * 3, bias=False)
        self.proj = nn.Linear(d_embed, d_embed, bias=False)

    def forward(self, x):

        batch_size, n_tokens, d_embed = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, n_tokens, self.n_heads, 3 * self.attn_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        residual = nn.functional.scaled_dot_product_attention(q, k, v)
        residual = residual.permute(0, 2, 1, 3)
        residual = residual.reshape(batch_size, n_tokens, self.n_heads * self.attn_dim)
        
        return self.proj(residual)