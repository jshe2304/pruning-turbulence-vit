import torch

def sinusoidal_embedding_1d(d_embed: int, n_positions: int):
    """
    Generate 1D sinusoidal embeddings.

    Args:
        d_embed: Output dimension for each position (must be even).
        n_positions: Number of positions to encode.

    Returns:
        Tensor of shape (M, d_embed) containing sinusoidal embeddings.
    """
    assert d_embed % 2 == 0, "d_embed must be even."

    i = torch.arange(d_embed // 2)
    omega = 10000 ** (-2 * i / d_embed)
    pos = torch.arange(n_positions)
    angles = pos.unsqueeze(1) * omega.unsqueeze(0)

    emb_sin = torch.sin(angles)
    emb_cos = torch.cos(angles)

    emb = torch.cat([emb_sin, emb_cos], dim=1)

    return emb

def sinusoidal_embedding_3d(d_embed: int, grid_size: tuple[int, int, int]):
    """
    Generate 3D sinusoidal embeddings for a (t, h, w) grid.

    Args:
        d_embed: Total output dimension (must be divisible by 16).
        grid_size: Tuple (t_size, h_size, w_size).

    Returns:
        Tensor of shape (t*h*w, d_embed) containing 3D sinusoidal embeddings.
    """

    assert d_embed % 16 == 0, "d_embed must be divisible by 16."

    T, H, W = grid_size

    # Allocate dimensions
    w_dim = d_embed // 16 * 6
    h_dim = d_embed // 16 * 6
    t_dim = d_embed // 16 * 4

    # Generate 1D embeddings
    w_embed = sinusoidal_embedding_1d(w_dim, W)  # (w_size, w_dim)
    h_embed = sinusoidal_embedding_1d(h_dim, H)  # (h_size, h_dim)
    t_embed = sinusoidal_embedding_1d(t_dim, T)  # (t_size, t_dim)

    # Tile for full 3D grid
    # w repeated for each t and h
    w_full = w_embed.repeat(T * H, 1)  # (t*h*w, w_dim)

    # h: repeat each h entry w_size times, then tile for each t
    h_full = h_embed.unsqueeze(1).repeat(1, W, 1)
    h_full = h_full.view(-1, h_dim).repeat(T, 1)  # (t*h*w, h_dim)

    # t: repeat each t entry for each h*w location
    t_full = t_embed.repeat_interleave(H * W, dim=0)  # (t*h*w, t_dim)

    # Concatenate all dimensions
    return torch.cat([w_full, h_full, t_full], dim=1)  # (t*h*w, d_embed)
