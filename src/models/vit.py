import torch
import torch.nn as nn

from .transformer import TransformerBlock
from .embeddings import PatchEmbed
from .positional_encodings import sinusoidal_embedding_2d, sinusoidal_embedding_3d
from .conv import SubPixelConv2D, SubPixelConv3D

class ViT(nn.Module):

    def __init__(
        self, 
        d_embed: int = 192, 
        n_heads: int = 6, 
        n_layers: int = 6, 
        img_shape: tuple[int, int] = (256, 256), 
        patch_shape: tuple[int, int] = (16, 16)
    ):
        super().__init__()

        self.img_shape = img_shape
        self.patch_shape = patch_shape
        self.grid_shape = (img_shape[0] // patch_shape[0], img_shape[1] // patch_shape[1])

        self.patch_embed = PatchEmbed(
            img_shape=img_shape, 
            patch_shape=patch_shape, 
            in_channels=1,
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
            out_channels=1, 
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x += self.pos_embed
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        x = self.upsample(x)
        return x

class EncoderDecoderViT(nn.Module):

    def __init__(
        self,
        in_channels=1, 
        img_shape=(1, 256, 256),
        patch_shape=(1, 16, 16), 
        d_encoder_embed=192, 
        n_encoder_heads=6, 
        n_encoder_layers=6, 
        d_decoder_embed=96, 
        n_decoder_heads=6, 
        n_decoder_layers=6, 
    ):
        super().__init__()

        # Image shape and patch shape

        self.img_shape = img_shape
        self.patch_shape = patch_shape
        self.grid_shape = (
            img_shape[0] // patch_shape[0],
            img_shape[1] // patch_shape[1],
            img_shape[2] // patch_shape[2],
        )

        # Patch embedding

        self.patch_embed = PatchEmbed(
            img_shape=img_shape, 
            patch_shape=patch_shape, 
            in_channels=in_channels, 
            out_channels=d_encoder_embed,
        )

        # Encoder

        self.encoder_pos_embed = sinusoidal_embedding_3d(
            d_encoder_embed, self.grid_shape
        )
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(d_encoder_embed, n_encoder_heads)
            for i in range(n_encoder_layers)
        ])
        self.norm = nn.LayerNorm(d_encoder_embed)

        # Down projection to decoder space

        self.down_proj= nn.Linear(d_encoder_embed, d_decoder_embed, bias=True)

        # Decoder

        self.decoder_pos_embed = sinusoidal_embedding_3d(
            d_decoder_embed, self.grid_shape
        )
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(d_decoder_embed, n_decoder_heads)
            for i in range(n_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_decoder_embed)

        # Patch upsampling

        self.upsample = SubPixelConv3D(
            (1, img_shape[1], img_shape[2]),
            patch_shape,
            d_decoder_embed,
            in_channels,
        )

        # Initialize weights

        self.apply(self._init_weights)

    def _init_weights(self, m):
      if isinstance(m, nn.Linear):
          torch.nn.init.xavier_uniform_(m.weight)
          if isinstance(m, nn.Linear) and m.bias is not None:
              nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.LayerNorm):
           nn.init.constant_(m.bias, 0)
           nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of shape (B, C, T, H, W)

        Returns:
            x: Encoded tensor of shape (B, E, T, H, W)
        """
        x = self.patch_embed(x)
        x += self.encoder_pos_embed
        for blk in self.encoder_blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward_decoder(self, x):
        """
        Forward pass through the decoder.

        Args:
            x: Input tensor of shape (B, E, T, H, W)

        Returns:
            x: Decoded tensor of shape (B, C, T, H, W)
        """

        x = self.down_proj(x)
        x += self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        x = x.reshape(x.size(0), -1, *self.grid_shape)
        x = self.upsample(x)
        
        return x

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (B, C, T, H, W)

        Returns:
            x: Decoded tensor of shape (B, C, T, H, W)
        """

        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent)
        return pred


        
            