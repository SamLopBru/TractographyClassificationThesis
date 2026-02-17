
import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE).
    
    Encodes relative positions by rotating Q/K vectors using sinusoidal
    frequencies. Unlike absolute PE, this is applied inside each attention
    layer to the query and key projections.
    
    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
               (Su et al., 2021)
    """
    
    def __init__(self, d_head: int, max_len: int = 5000):
        super().__init__()
        assert d_head % 2 == 0, f"d_head must be even for RoPE, got {d_head}"
        
        # Precompute frequency bands: theta_i = 1 / 10000^(2i/d)
        freqs = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2).float() / d_head))
        positions = torch.arange(max_len).float()
        # (max_len, d_head/2)
        angles = torch.outer(positions, freqs)
        # Store as complex exponentials for efficient rotation
        # (max_len, d_head/2) complex
        self.register_buffer('cos_cached', angles.cos())
        self.register_buffer('sin_cached', angles.sin())
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply rotary encoding to input tensor.
        
        Args:
            x: Tensor of shape (..., seq_len, d_head)
            offset: Position offset (for cached KV in generation)
        
        Returns:
            Rotated tensor of same shape
        """
        seq_len = x.shape[-2]
        cos = self.cos_cached[offset:offset + seq_len]  # (seq_len, d_head/2)
        sin = self.sin_cached[offset:offset + seq_len]  # (seq_len, d_head/2)
        
        # Split into pairs and rotate
        x1 = x[..., 0::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices
        
        # Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)
        
        return rotated