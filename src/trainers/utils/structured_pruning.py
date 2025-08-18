from re import L
import torch
from .compute_loss import compute_loss

@torch.no_grad()
def prune_attention_head(model, dataset, device):
    """
    Prune the attention head of the model.
    """

    blocks = model.module.encoder_blocks + model.module.decoder_blocks

    min_loss = float('inf')
    layer, head_index = None, None

    for l in range(len(blocks)):
        attn = blocks[l].attn
        for h in range(attn.num_heads):

            # Get and shape masks

            qkv_mask = attn.qkv.weight_mask.T
            qkv_mask = qkv_mask.view(attn.embed_dim, attn.num_heads, 3 * attn.attn_dim)

            qkv_bias_mask = attn.qkv.bias_mask
            qkv_bias_mask = qkv_bias_mask.view(attn.num_heads, 3 * attn.attn_dim)

            proj_mask = attn.proj.weight_mask.T
            proj_mask = proj_mask.view(attn.num_heads, attn.attn_dim, attn.embed_dim)

            # Continue if head is already fully pruned

            if not torch.any(qkv_mask[:, h]) and not torch.any(proj_mask[h]): continue

            # Prune head, compute loss, and restore mask

            qkv_mask[:, h] = 0
            qkv_bias_mask[h] = 0
            proj_mask[h] = 0
            loss = compute_loss(model, dataset, device=device)
            qkv_mask[:, h] = 1
            qkv_bias_mask[h] = 1
            proj_mask[h] = 1

            # Update best loss and layer/head index

            if loss < min_loss:
                min_loss = loss
                layer, head_index = l, h

    # Permanently prune head

    attn = blocks[layer].attn
    
    qkv_mask = attn.qkv.weight_mask.T
    qkv_mask = qkv_mask.view(attn.embed_dim, attn.num_heads, 3 * attn.attn_dim)
    qkv_mask[:, head_index] = 0

    qkv_bias_mask = attn.qkv.bias_mask
    qkv_bias_mask = qkv_bias_mask.view(attn.num_heads, 3 * attn.attn_dim)
    qkv_bias_mask[head_index] = 0

    proj_mask = attn.proj.weight_mask.T
    proj_mask = proj_mask.view(attn.num_heads, attn.attn_dim, attn.embed_dim)
    proj_mask[head_index] = 0

    return layer, head_index
