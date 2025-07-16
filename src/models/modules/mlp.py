import torch
import torch.nn as nn
from math import sqrt

class MLP(nn.Module):
    """
    Multi-layer perceptron with latent space pruning. 
    """
    def __init__(self, d_embed, expansion_factor=4, dropout=0.1):
        super().__init__()

        d_latent = d_embed * expansion_factor

        self.W_up = nn.Parameter(torch.empty(d_embed, d_latent))
        self.W_down = nn.Parameter(torch.empty(d_latent, d_embed))

        self.bias_up = nn.Parameter(torch.empty(d_latent))
        self.bias_down = nn.Parameter(torch.empty(d_embed))

        self.latent_mask = torch.ones(d_latent, dtype=torch.bool)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        
        # Initialize weights

        nn.init.kaiming_uniform_(self.W_up, a=sqrt(5))
        nn.init.kaiming_uniform_(self.W_down, a=sqrt(5))

        # Initialize biases

        fan_in_up, _ = nn.init._calculate_fan_in_and_fan_out(self.W_up)
        bound_up = 1 / sqrt(fan_in_up) if fan_in_up > 0 else 0
        nn.init.uniform_(self.bias_up, -bound_up, bound_up)

        fan_in_down, _ = nn.init._calculate_fan_in_and_fan_out(self.W_down)
        bound_down = 1 / sqrt(fan_in_down) if fan_in_down > 0 else 0
        nn.init.uniform_(self.bias_down, -bound_down, bound_down)

    def pruned_mlp_matrices(self):
        """
        Returns pruned MLP matrices
        """

        W_up = self.W_up[:, self.latent_mask]
        bias_up = self.bias_up[self.latent_mask]
        W_down = self.W_down[self.latent_mask, :]

        return W_up, bias_up, W_down, self.bias_down

    def forward(self, e):
        """
        Input:
            e: Embeddings of shape (n_batch, n_tokens, d_embed)
        
        Output:
            e: Updated embeddings of shape (n_batch, n_tokens, d_embed)
        """

        # Get pruned MLP matrices

        W_up, bias_up, W_down, bias_down = self.pruned_mlp_matrices()

        e = e @ W_up + bias_up
        e = self.activation(e)
        e = self.dropout(e)
        e = e @ W_down + bias_down
        e = self.dropout(e)

        return e
