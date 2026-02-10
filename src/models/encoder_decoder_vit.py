# Code adpated from IBM/NASA's Prithvi

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from einops import rearrange

from .modules.attention import Attention
from .modules.mlp import MLP
from .modules.embeddings import PatchEmbed
from .modules.positional_encodings import get_3d_sincos_pos_embed
from .modules.conv import SubPixelConvICNR_3D

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, qkv_bias=False, proj_bias=True):

        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(
            embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class EncoderDecoderViT(nn.Module):
    """
    Vision Transformer
    """
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        num_frames=1,
        tubelet_size=1,
        in_chans=1,
        encoder_embed_dim=192,
        encoder_depth=6,
        encoder_num_heads=6,
        decoder_embed_dim=192,
        decoder_depth=6,
        decoder_num_heads=6,
        mlp_ratio=4., 
        num_out_frames=1,
        patch_recovery='linear', # ['linear',conv','subpixel_conv']
        **kwargs
    ):
        super().__init__()

        # Encoder

        self.patch_embed = PatchEmbed(img_size, patch_size, num_frames, tubelet_size, in_chans,
                                    encoder_embed_dim, nn.LayerNorm)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, encoder_embed_dim), requires_grad=False)

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True)
            for i in range(encoder_depth)
        ])
        self.norm = nn.LayerNorm(encoder_embed_dim)

        # Decoder

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True)
            for i in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Patch recovery

        self.patchrecovery = SubPixelConvICNR_3D(
            (num_out_frames,img_size,img_size), 
            (tubelet_size,patch_size,patch_size), 
            decoder_embed_dim, in_chans
        )
        self.patch_recovery = 'subpixel_conv'
        self.num_out_frames = num_out_frames

        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
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

    def get_weights(self):
        weights = []

        for block in self.encoder_blocks:
            weights += [
                (block.attn.qkv, 'weight'), 
                (block.attn.proj, 'weight'), 
                (block.mlp.fc1, 'weight'),
                (block.mlp.fc2, 'weight'), 
            ]

        weights.append((self.decoder_embed, 'weight'))

        for block in self.decoder_blocks:
            weights += [
                (block.attn.qkv, 'weight'), 
                (block.attn.proj, 'weight'), 
                (block.mlp.fc1, 'weight'),
                (block.mlp.fc2, 'weight'), 
            ]
        
        return weights

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """ For calculating loss.
        imgs: B, C, T, H, W
        x: B, L, D
        """
        p = self.patch_embed.patch_size[0]
        tub = self.num_out_frames
        x = rearrange(imgs, 'b c (t tub) (h p) (w q) -> b (t h w) (tub p q c)',
                    tub=tub, p=p, q=p)
        return x

    def unpatchify(self, x):
        """ For calculating loss.
        x: B, L, D
        imgs: B, C, T, H, W
        """
        p = self.patch_embed.patch_size[0]
        num_p = self.patch_embed.img_size[0] // p
        tub = self.num_out_frames
        imgs = rearrange(x, 'b (t h w) (tub p q c) -> b c (t tub) (h p) (w q)',
                        h=num_p, w=num_p, tub=tub, p=p, q=p)
        return imgs

    def decoder_pred(self, x):
        if isinstance(self.patchrecovery, nn.Linear):
            x = self.patchrecovery(x)
            x = self.unpatchify(x)
            return x
        else:
            # reshape: [B, L, D] -> [B, C, num_patches_T, num_patches_X, num_patches_Y]
            B, _, _ = x.shape
            t, h, w = self.patch_embed.grid_size
            x = x.reshape(B, -1, t, h, w)
            x = self.patchrecovery(x)
            return x

    def forward_encoder(self, x, train=False):
        # embed patches + add position encoding
        x = self.patch_embed(x)
        x = x + self.pos_embed

        for blk in self.encoder_blocks:
            x = blk(x)

        x = self.norm(x)

        return x

    def forward_decoder(self, x, train=False):
        # embed tokens + add position encoding
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, img, pred, weights=None):
        """
        img: B, C, T, H, W
        pred: B, C, T, H, W
        weights: (optional) B, C, T, H, W
        """

        loss = (pred - img) ** 2
        if weights is not None:
            loss *= weights
        #loss = torch.abs(pred-img)
        loss = loss.mean()

        return loss 

    def spectral_loss(self, img, pred, weight, threshold_wavenumber):
        """
        img: B, C, T, H, W
        pred: B, C, T, H, W
        """
        # Calculating zonal fft and averageing
        img_hat = torch.mean(torch.abs(torch.fft.rfft(img,dim=3)),dim=4)
        pred_hat = torch.mean(torch.abs(torch.fft.rfft(pred,dim=3)),dim=4)

        # Loss for both channels
        loss1 = (pred_hat[:,0,:,threshold_wavenumber:]-img_hat[:,0,:,threshold_wavenumber:]) ** 2
        loss2 = (pred_hat[:,1,:,threshold_wavenumber:]-img_hat[:,1,:,threshold_wavenumber:]) ** 2

        loss = weight*0.5*(loss1.mean() + loss2.mean())

        return loss

    def forward(self, x, train=False):
        latent = self.forward_encoder(x, train=train)
        pred = self.forward_decoder(latent, train=train)
        #pred = self.unpatchify(pred)

        return pred

    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        """
        Load a checkpoint into the model. 
        Handles pruning buffers and distributed prefixes.

        Args:
            state_dict (dict)
            strict (bool): Enforce keys in `state_dict` match the model's keys?
            *args, **kwargs: Overflow
        """

        # Extract model state

        state_dict = getattr(state_dict, 'model_state', state_dict)

        # If there are pruning buffers, set up pruning
        for key in list(state_dict):
            if key.endswith('_orig') or key.endswith('_mask'):
                submodule_name, buffer_name = key.rsplit('.', 1)
                param_name = buffer_name.rsplit('_', 1)[0]

                prune.identity(self.get_submodule(submodule_name), param_name)

        # Strip "module." prefix from distributed checkpoints

        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }

        # Delegate to parent implementation

        super().load_state_dict(state_dict, strict=strict, *args, **kwargs)
