import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from .modules.attention import Attention
from .modules.mlp import MLP
from .modules.embeddings import PatchEmbed
from .modules.positional_encodings import get_3d_sincos_pos_embed
from .modules.conv import SubPixelConvICNR_3D


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, qkv_bias=False, proj_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), out_features=embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ReynoldsEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, reynolds):
        """
        reynolds: (B,) scalar per sample
        returns: (B, 1, embed_dim)
        """
        re_log = torch.log(reynolds).unsqueeze(-1)  # (B, 1)
        return self.mlp(re_log).unsqueeze(1)  # (B, 1, embed_dim)


class ReSimpleViT(nn.Module):
    """
    Encoder-only Vision Transformer with Reynolds number conditioning.
    Reynolds number is embedded via a small MLP and added to token embeddings.
    """
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=6,
        num_frames=1,
    ):
        super().__init__()

        in_chans = 2
        tubelet_size = num_frames
        mlp_ratio = 4.

        self.patch_embed = PatchEmbed(
            img_size, patch_size, num_frames, tubelet_size, in_chans,
            embed_dim, nn.LayerNorm
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )

        self.re_embed = ReynoldsEmbedding(embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.patchrecovery = SubPixelConvICNR_3D(
            (1, img_size, img_size),
            (tubelet_size, patch_size, patch_size),
            embed_dim, in_chans
        )

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_pruned_parameters(self):
        pruned = 0
        for module in self.modules():
            for buffer_name, buffer in module.named_buffers(recurse=False):
                if buffer_name.endswith('_mask'):
                    pruned += int((buffer == 0).sum().item())
        return pruned

    def get_weights(self):
        weights = []
        for block in self.blocks:
            weights += [
                (block.attn.qkv, 'weight'),
                (block.attn.proj, 'weight'),
                (block.mlp.fc1, 'weight'),
                (block.mlp.fc2, 'weight'),
            ]
        return weights

    def forward_loss(self, img, pred, weights=None):
        loss = (pred - img) ** 2
        if weights is not None:
            loss *= weights
        return loss.mean()

    def forward(self, x, reynolds, train=False):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = x + self.re_embed(reynolds)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        B = x.shape[0]
        t, h, w = self.patch_embed.grid_size
        x = x.reshape(B, -1, t, h, w)
        x = self.patchrecovery(x)
        return x

    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        state_dict = getattr(state_dict, 'model_state', state_dict)
        for key in list(state_dict):
            if key.endswith('_orig') or key.endswith('_mask'):
                submodule_name, buffer_name = key.rsplit('.', 1)
                param_name = buffer_name.rsplit('_', 1)[0]
                prune.identity(self.get_submodule(submodule_name), param_name)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        super().load_state_dict(state_dict, strict=strict, *args, **kwargs)
