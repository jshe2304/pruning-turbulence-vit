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
        in_channels: int, 
        out_channels: int, 
        img_shape: tuple[int, int], 
        patch_size: int, 
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
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels, img_shape, patch_shape):
        super().__init__()
        self.img_shape = img_shape
        self.patch_shape = patch_shape
        assert patch_shape[1] == patch_shape[2], 'mismatch'
        self.in_chans_per_frame = in_channels // patch_shape[0]  # patch_size is tubelet size
        self.conv = nn.Conv2d(self.in_chans_per_frame, out_channels*patch_shape[1]**2, kernel_size=3, stride=1, padding=1, bias=0, padding_mode='circular')
        self.pixelshuffle = nn.PixelShuffle(patch_shape[1])
        self.conv.weight.data.copy_(initialize_icnr(
            self.conv.weight,   
            initializer=nn.init.kaiming_normal_,
            upscale_factor=patch_shape[1]
        ))

    def forward(self, x: torch.Tensor):
        # first, split in dimension
        # print(x.shape)
        x = x.reshape(x.shape[0], self.in_chans_per_frame, self.patch_shape[0], *x.shape[2:]).flatten(2, 3)[:, :, 0:self.img_shape[0]] # to make 13 vertical dims
        #print(x.shape)
        x = x.movedim(-3, 1).flatten(0, 1)
        output = self.conv(x)
        output = self.pixelshuffle(output)
        #print(output.shape)
        output = output.reshape(-1, self.img_size[0], *output.shape[1:]).movedim(1, -3)

        _, _, Pl, Lat, Lon = output.shape

        pl_pad = Pl - self.img_shape[0]
        lat_pad = Lat - self.img_shape[1]
        lon_pad = Lon - self.img_shape[2]

        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front

        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top

        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

        return output[:, :, padding_front: Pl - padding_back,
               padding_top: Lat - padding_bottom, padding_left: Lon - padding_right]

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
    