import torch
import torch.nn as nn

def initialize_icnr(tensor, initializer, upscale_factor=2, *args, **kwargs):
    """
    Initialize the weights of the convolutional layer using the ICNR method.

    Args:
        tensor: >2-dimensional Tensor
        initializer: the initialization method
        upscale_factor: the upscale factor
    """

    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0

    # Construct the sub_kernel and initialize
    sub_kernel = torch.empty(
        tensor.shape[0] // upscale_factor_squared,
        *tensor.shape[1:]
    )
    sub_kernel = initializer(sub_kernel, *args, **kwargs)

    # Repeat the sub_kernel to the original size
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)

class SubPixelConv2D(nn.Module):
    """
    Patch Embedding Recovery to 2D Image.

    Args:
        img_shape (tuple[int]): Lat, Lon
        patch_size (int): Lat, Lon
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(
        self, 
        img_shape: tuple[int, int], 
        patch_size: int, 
        in_channels: int, 
        out_channels: int,
    ):
        super().__init__()

        self.img_shape = img_shape

        self.conv = nn.Conv2d(
            in_channels, out_channels * patch_size ** 2, 
            kernel_size=3, stride=1, padding=1, 
            bias=False, padding_mode='circular'
        )
        self.conv.weight.data.copy_(initialize_icnr(
            self.conv.weight,   
            initializer=nn.init.kaiming_normal_,
            upscale_factor=patch_size
        ))
        self.pixel_shuffle = nn.PixelShuffle(patch_size)

    def forward(self, x: torch.Tensor):

        # Apply conv layer
        output = self.conv(x)

        # Apply pixel shuffle
        output = self.pixel_shuffle(output)

        # Pad output to match image shape
        _, _, H, W = output.shape
        h_pad = H - self.img_shape[0]
        w_pad = W - self.img_shape[1]

        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)

        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)

        # Return padded output
        return output[:, :, padding_top: H - padding_bottom, padding_left: W - padding_right]

class SubPixelConv3D(nn.Module):

    """
    Patch Embedding Recovery to 3D Image.

    Args:
        img_shape (tuple[int]): T, H, W
        patch_shape (tuple[int]): T, H, W
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_shape, patch_shape, in_channels, out_channels):
        super().__init__()

        assert patch_shape[1] == patch_shape[2], 'mismatch'
        patch_size = patch_shape[1]

        # Initialize the convolutional layer
        self.T, self.H, self.W = img_shape
        self.conv = nn.Conv2d(
            in_channels // 2, out_channels * patch_size ** 2, 
            kernel_size=3, stride=1, padding=1, 
            bias=False, padding_mode='circular'
        )
        self.conv.weight.data.copy_(initialize_icnr(
            self.conv.weight,   
            initializer=nn.init.kaiming_normal_,
            upscale_factor=patch_size
        ))

        # Initialize the pixel shuffle layer
        self.pixel_shuffle = nn.PixelShuffle(patch_size)

    def forward(self, x: torch.Tensor):

        # Manipulate input for conv layer
        x = x.reshape(
            x.shape[0], x.shape[1] // 2, 2, *x.shape[2:]
        ) # B, C//2, 2, T, H, W
        x = x.flatten(2, 3) # B, C//2, 2*T, H, W
        x = x[:, :, 0:self.T] # B, C//2, T, H, W
        x = x.movedim(-3, 1) # B, T, C//2, H, W
        x = x.flatten(0, 1) # B*T, C//2, H, W

        # Apply conv layer
        output = self.conv(x) # B*T, C*H*W, 1, 1
        output = self.pixel_shuffle(output)
        output = output.reshape(
            -1, self.T, *output.shape[1:]
        ).movedim(1, -3)

        _, _, T, W, H = output.shape

        t_pad = T - self.T
        w_pad = W - self.W
        h_pad = H - self.H

        padding_t0 = t_pad // 2
        padding_t1 = t_pad - padding_t0

        padding_w0 = w_pad // 2
        padding_w1 = w_pad - padding_w0

        padding_h0 = h_pad // 2
        padding_h1 = h_pad - padding_h0

        return output[
            :, :, 
            padding_t0: T - padding_t1,
            padding_w0: W - padding_w1, 
            padding_h0: H - padding_h1
        ]

if __name__ == '__main__':

    # Test 2D subpixel conv

    B, H, W, E = 32, 64, 64, 128
    x = torch.randn(B, E, H, W)
    
    subpixel_conv = SubPixelConv2D(
        img_shape=(256, 256), patch_size=4, 
        in_channels=128, out_channels=2
    )

    x = subpixel_conv(x)
    print(f'2D subpixel conv shape: {x.shape}')
    