import torch
import torch.nn as nn

from .modules.embeddings import PatchEmbed2D
from .modules.positional_encodings import sinusoidal_embedding_2d
from .modules.conv import SubPixelConv2D
from .modules.transformer_block import TransformerBlock

class ViT(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        d_embed: int, 
        n_heads: int, 
        n_layers: int, 
        img_shape: tuple[int, int], 
        patch_shape: tuple[int, int],
        **kwargs,
    ):
        super().__init__()

        self.img_shape = img_shape
        self.patch_shape = patch_shape
        self.grid_shape = (img_shape[0] // patch_shape[0], img_shape[1] // patch_shape[1])

        self.patch_embed = PatchEmbed2D(
            patch_shape=patch_shape, 
            in_channels=in_channels,
            out_channels=d_embed,
        )

        self.pos_embed = sinusoidal_embedding_2d(d_embed, *self.grid_shape)

        # Transformer blocks and coalesced head/latent masks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_embed, n_heads)
            for i in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_embed)

        self.upsample = SubPixelConv2D(
            img_shape=img_shape, 
            patch_size=patch_shape[0], 
            in_channels=d_embed, 
            out_channels=in_channels, 
        )

    def get_parameters_to_prune(self):
        """
        Returns transformer block parameters for global pruning. 
        """
        parameters_to_prune = []

        for block in self.transformer_blocks:
            parameters_to_prune += [
                (block.attn, 'QKV'), 
                (block.attn, 'out_proj'), 
                (block.mlp, 'W_up'), 
                (block.mlp, 'W_down'), 
                (block.mlp, 'bias_up'),
                (block.mlp, 'bias_down'),
            ]
            
        return parameters_to_prune

    def n_parameters(self):
        """
        Returns the number of parameters in the model. 
        """
        return sum(p.numel() for p in self.parameters())

    def n_pruned_parameters(self):
        """
        Returns the number of unpruned parameters in the model. 
        """

        pruned = 0
        for module in self.modules():
            for buffer_name, buffer in module.named_buffers(recurse=False):
                if buffer_name.endswith('_mask'):
                    pruned += int((buffer == 0).sum().item())

        return pruned

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            x: Output tensor of shape (B, C, H, W)
        """

        B, *_ = x.shape

        x = self.patch_embed(x)
        x += self.pos_embed.to(x.device)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        x = x.reshape(B, -1, *self.grid_shape)
        x = self.upsample(x)
        return x

        
            