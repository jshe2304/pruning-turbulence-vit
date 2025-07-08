import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    Patch Embedding module for 3D/temporal images. 
    """

    def __init__(
        self,
        img_shape, patch_shape, 
        in_channels, out_channels, 
    ):
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
