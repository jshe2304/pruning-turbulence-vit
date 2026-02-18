import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode='circular'),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode='circular'),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    2D UNet for turbulence prediction.
    Matches the ViT interface: input/output shape (B, C, T, H, W).
    Frames are folded into the channel dimension for 2D convolution.
    """
    def __init__(
        self,
        img_size=256,
        num_frames=1,
        base_channels=64,
        depth=4,
        dropout=0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_frames = num_frames
        in_chans = 2 * num_frames
        out_chans = 2 * num_frames

        # Encoder
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]
        self.inc = DoubleConv(in_chans, channels[0], dropout)
        self.downs = nn.ModuleList([
            Down(channels[i], channels[i + 1], dropout) for i in range(depth)
        ])

        # Decoder
        self.ups = nn.ModuleList([
            Up(channels[i + 1], channels[i], dropout) for i in range(depth - 1, -1, -1)
        ])
        self.outc = nn.Conv2d(channels[0], out_chans, 1)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                weights.append((module, 'weight'))
        return weights

    def forward_loss(self, img, pred, weights=None):
        loss = (pred - img) ** 2
        if weights is not None:
            loss *= weights
        return loss.mean()

    def forward(self, x, *args, **kwargs):
        B, C, T, H, W = x.shape
        # Fold frames into channel dim: (B, C*T, H, W)
        x = x.reshape(B, C * T, H, W)

        # Encoder with skip connections
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # Decoder (skip the bottleneck, reverse the rest)
        skips = skips[:-1][::-1]
        for up, skip in zip(self.ups, skips):
            x = up(x, skip)

        x = self.outc(x)

        # Unfold back: (B, C, T, H, W)
        x = x.reshape(B, C, T, H, W)
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
