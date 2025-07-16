import torch
import torch.nn as nn

class PatchEmbed2D(nn.Module):
    """
    Patch Embedding module for 2D images. 
    """

    def __init__(self, patch_shape, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=patch_shape,
            stride=patch_shape,
        )
        w = self.conv.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        """
        Input:
            x: Batched 2D images of shape (B, C, H, W)
        Output:
            x: Batched embeddings of shape (B, L, C)
        """

        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class PatchEmbed3D(nn.Module):
    """
    Patch Embedding module for 3D images. 
    """

    def __init__(self, patch_shape, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=patch_shape,
            stride=patch_shape,
        )
        w = self.conv.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        """
        Input:
            x: Batched 3D/temporal images of shape (B, C, T, H, W)
        Output:
            x: Batched embeddings of shape (B, L, C)
        """

        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

if __name__ == '__main__':

    # Test 2D patch embedding

    B, C, H, W = 16, 2, 128, 128
    x = torch.randn(B, C, H, W)

    patch_embed = PatchEmbed2D(patch_shape=(4, 4), in_channels=2, out_channels=128)
    print('Initialized 2D patch embedding.')
    x = patch_embed(x)
    print(f'2D patch embedding shape: {x.shape}')
    