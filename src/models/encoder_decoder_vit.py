import torch
import torch.nn as nn

from .modules.embeddings import PatchEmbed3D
from .modules.positional_encodings import sinusoidal_embedding_3d
from .modules.conv import SubPixelConv3D
from .modules.transformer_block import TransformerBlock

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

        self.patch_embed = PatchEmbed3D(
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

