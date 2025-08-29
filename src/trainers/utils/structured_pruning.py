import torch
from .compute_loss import compute_loss
import torch.nn.utils.prune as prune

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

            # Create pruning buffers if needed

            if not hasattr(attn.qkv, 'weight_mask'):
                prune.identity(attn.qkv, 'weight')
            if not hasattr(attn.qkv, 'bias_mask'):
                prune.identity(attn.qkv, 'bias')
            if not hasattr(attn.proj, 'weight_mask'):
                prune.identity(attn.proj, 'weight')

            # Get and shape masks

            qkv_mask = attn.qkv.weight_mask.T.reshape(attn.embed_dim, 3, attn.num_heads, attn.attn_dim)
            qkv_bias_mask = attn.qkv.bias_mask.reshape(3, attn.num_heads, attn.attn_dim)
            proj_mask = attn.proj.weight_mask.T.reshape(attn.num_heads, attn.attn_dim, attn.embed_dim)

            # Continue if head is already fully pruned

            head_qkv_zero = not torch.any(qkv_mask[:, :, h])
            head_bias_zero = not torch.any(qkv_bias_mask[:, h])
            head_proj_zero = not torch.any(proj_mask[h])
            if head_qkv_zero and head_proj_zero and head_bias_zero: continue

            # Cache mask slices

            prev_qkv_slice = qkv_mask[:, :, h].clone()
            prev_bias_slice = qkv_bias_mask[:, h].clone()
            prev_proj_slice = proj_mask[h].clone()
            
            # Prune head

            qkv_mask[:, :, h] = 0
            qkv_bias_mask[:, h] = 0
            proj_mask[h] = 0

            # Compute loss

            loss = compute_loss(model, dataset, num_rollout_steps=1, device=device)

            # Restore mask slices

            qkv_mask[:, :, h] = prev_qkv_slice
            qkv_bias_mask[:, h] = prev_bias_slice
            proj_mask[h] = prev_proj_slice

            # Update best loss and layer/head index

            if loss < min_loss:
                min_loss = loss
                layer, head_index = l, h

    # Permanently prune head

    attn = blocks[layer].attn
    
    qkv_mask = attn.qkv.weight_mask.T.reshape(attn.embed_dim, 3, attn.num_heads, attn.attn_dim)
    qkv_mask[:, :, head_index] = 0

    qkv_bias_mask = attn.qkv.bias_mask.reshape(3, attn.num_heads, attn.attn_dim)
    qkv_bias_mask[:, head_index] = 0

    proj_mask = attn.proj.weight_mask.T.reshape(attn.num_heads, attn.attn_dim, attn.embed_dim)
    proj_mask[head_index] = 0

    return layer, head_index

def num_pruned_heads(model):
    """
    Count the number of pruned heads in the model.
    """

    model = getattr(model, 'module', model)
    blocks = model.encoder_blocks + model.decoder_blocks

    num_pruned = 0
    for l, block in enumerate(blocks):
        attn = block.attn
        for h in range(attn.num_heads):

            # Get and shape masks

            qkv_mask = attn.qkv.weight_mask.T.reshape(attn.embed_dim, 3, attn.num_heads, attn.attn_dim)
            qkv_bias_mask = attn.qkv.bias_mask.reshape(3, attn.num_heads, attn.attn_dim)
            proj_mask = attn.proj.weight_mask.T.reshape(attn.num_heads, attn.attn_dim, attn.embed_dim)

            # Continue if head is already fully pruned

            head_qkv_zero = not torch.any(qkv_mask[:, :, h])
            head_bias_zero = not torch.any(qkv_bias_mask[:, h])
            head_proj_zero = not torch.any(proj_mask[h])
            if head_qkv_zero and head_bias_zero and head_proj_zero:
                print(f"Pruned head {h} in block {l}")
                num_pruned += 1

    return num_pruned
