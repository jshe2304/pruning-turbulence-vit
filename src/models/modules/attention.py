import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, proj_bias=True):
        super().__init__()
        
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dim = embed_dim // num_heads
        self.scale = self.attn_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=proj_bias)

    def forward(self, x):

        batch_size, n_tokens, embed_dim = x.shape

        # Compute and separate Q, K, V matrices

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.num_heads, self.attn_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Attention calculation

        x = nn.functional.scaled_dot_product_attention(q, k, v,)

        # Up projection back into embedding space

        x = x.transpose(1, 2)
        x = x.reshape(batch_size, n_tokens, embed_dim)
        x = self.proj(x)
        
        return x
