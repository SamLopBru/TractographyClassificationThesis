import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
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


class StreamlineEncoder(nn.Module):
    """
    Transformer-based encoder for classifying streamlines into bundles.
    
    Input: Streamlines of shape (batch_size, seq_len, 5) where 5 = (r_norm, theta_cos, theta_sin, phi_cos, phi_sin)
    Output: Class logits of shape (batch_size, num_classes)
    """
    
    def __init__(
        self, 
        input_size: int = 5,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        num_classes: int = 32,
        max_len: int = 5000,
        dropout: float = 0.1,
        pooling: str = 'cls'  # 'cls', 'mean', or 'max'
    ):
        """
        Args:
            input_size: Number of features per point (default 5 for spherical coords)
            d_model: Dimension of the transformer model
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            num_classes: Number of bundle classes to predict
            max_len: Maximum sequence length for positional encoding
            dropout: Dropout rate
            pooling: Pooling strategy ('cls' for CLS token, 'mean' for mean pooling, 'max' for max pooling)
        """
        super().__init__()
        
        self.d_model = d_model
        self.pooling = pooling
        
        # Input projection: (batch, seq, 5) -> (batch, seq, d_model)
        self.input_projection = nn.Linear(input_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # CLS token for classification (learnable)
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder with stochastic depth
        # Linearly increasing drop path rate from 0 to drop_path_rate
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerWithDropPath(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                drop_path=dpr[i],
                norm_first=True  # Pre-norm for better training stability
            )
            for i in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)  # Final norm for pre-norm architecture
        
        # Classification head with bottleneck
        bottleneck_dim = d_model // 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with targeted initialization."""
        for name, p in self.named_parameters():
            if 'cls_token' in name:
                nn.init.normal_(p, std=0.02)  # Small random init for CLS token
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.zeros_(p)  # Biases to zero
    
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor = None,
        padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, 5)
            lengths: Optional tensor of actual sequence lengths (batch_size,)
            padding_mask: Optional boolean mask where True indicates padding positions (batch_size, seq_len)
        
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # Create padding mask from lengths if not provided
        if padding_mask is None and lengths is not None:
            # Create mask: True where positions are padding (to be ignored)
            indices = torch.arange(seq_len, device=x.device).unsqueeze(0)
            padding_mask = indices >= lengths.unsqueeze(1)
        
        # Project input to model dimension with normalization
        x = self.input_norm(self.input_projection(x))  # (batch, seq, d_model)
        
        # Add CLS token if using cls pooling
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1 + seq, d_model)
            
            # Extend padding mask for CLS token (CLS is never masked)
            if padding_mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        
        # Transpose for positional encoding: (batch, seq, d_model) -> (seq, batch, d_model)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # Back to (batch, seq, d_model)
        
        # Apply transformer encoder layers with stochastic depth
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=padding_mask)
        x = self.final_norm(x)
        
        # Pool the sequence
        if self.pooling == 'cls':
            # Use CLS token representation
            pooled = x[:, 0]  # (batch, d_model)
        elif self.pooling == 'mean':
            # Mean pooling (excluding padding)
            if padding_mask is not None:
                mask = ~padding_mask.unsqueeze(-1)  # (batch, seq, 1)
                x = x * mask
                pooled = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)
        elif self.pooling == 'max':
            # Max pooling (excluding padding)
            if padding_mask is not None:
                x = x.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
            pooled = x.max(dim=1)[0].clamp(min=-1e9)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits


class LightweightStreamlineEncoder(nn.Module):
    """
    A lighter LSTM-based encoder for faster training/inference.
    
    Good for initial experiments or resource-constrained settings.
    Supports multiple pooling strategies: 'last', 'cls', 'mean', 'max'.
    """
    
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 32,
        dropout: float = 0.1,
        bidirectional: bool = True,
        pooling: str = 'last'  # 'last', 'cls', 'mean', or 'max'
    ):
        """
        Args:
            input_size: Number of features per point (default 5 for spherical coords)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of bundle classes to predict
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            pooling: Pooling strategy:
                - 'last': Use final hidden state (original behavior)
                - 'cls': Prepend learnable CLS token and use its output
                - 'mean': Mean pooling over all timesteps
                - 'max': Max pooling over all timesteps
        """
        super().__init__()
        
        self.pooling = pooling
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.output_size = hidden_size * self.num_directions
        
        # CLS token (learnable, in input space)
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, input_size))
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        bottleneck_dim = hidden_size  # output_size -> hidden_size -> num_classes
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.output_size),
            nn.Dropout(dropout),
            nn.Linear(self.output_size, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, num_classes)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor = None,
        padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            lengths: Optional tensor of actual sequence lengths (batch_size,)
            padding_mask: Optional boolean mask where True indicates padding (batch_size, seq_len)
        
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # Create padding mask from lengths if not provided
        if padding_mask is None and lengths is not None:
            indices = torch.arange(seq_len, device=x.device).unsqueeze(0)
            padding_mask = indices >= lengths.unsqueeze(1)
        
        # Prepend CLS token if using cls pooling
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1 + seq, input_size)
            
            # Update lengths and mask for CLS token
            if lengths is not None:
                lengths = lengths + 1
            if padding_mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        
        # Process through LSTM
        if self.pooling == 'last':
            # Original behavior: use final hidden state
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                _, (hidden, _) = self.lstm(packed)
            else:
                _, (hidden, _) = self.lstm(x)
            
            # hidden: (num_layers * num_directions, batch, hidden_size)
            if self.num_directions == 2:
                pooled = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                pooled = hidden[-1]
        else:
            # Need all outputs for cls/mean/max pooling
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                output, _ = self.lstm(packed)
                output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            else:
                output, _ = self.lstm(x)
            
            # output: (batch, seq_len, hidden_size * num_directions)
            if self.pooling == 'cls':
                # Use the output corresponding to the CLS token (first position)
                pooled = output[:, 0]  # (batch, output_size)
            elif self.pooling == 'mean':
                # Mean pooling (excluding padding)
                if padding_mask is not None:
                    mask = ~padding_mask.unsqueeze(-1)  # (batch, seq, 1)
                    output = output * mask
                    pooled = output.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    pooled = output.mean(dim=1)
            elif self.pooling == 'max':
                # Max pooling (excluding padding)
                if padding_mask is not None:
                    output = output.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
                pooled = output.max(dim=1)[0].clamp(min=-1e9)
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        logits = self.classifier(pooled)
        return logits


# Convenience factory function
def create_encoder(
    encoder_type: str = 'transformer',
    num_classes: int = 32,
    **kwargs
) -> nn.Module:
    """
    Factory function to create streamline encoders.
    
    Args:
        encoder_type: 'transformer' or 'lstm'
        num_classes: Number of bundle classes
        **kwargs: Additional arguments passed to the encoder
    
    Returns:
        Encoder module
    """
    if encoder_type == 'transformer':
        return StreamlineEncoder(num_classes=num_classes, **kwargs)
    elif encoder_type == 'lstm':
        return LightweightStreamlineEncoder(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# if __name__ == "__main__":
#     from src.config import TrainConfig, DEFAULT_CONFIG

#     cfg = DEFAULT_CONFIG

#     # Test the encoders
#     batch_size = cfg.batch_size
#     seq_len = 150  # Variable length streamlines
#     input_size = cfg.input_size
#     num_classes = cfg.num_classes
    
#     # Create sample data
#     x = torch.randn(batch_size, seq_len, input_size)
#     lengths = torch.randint(50, seq_len + 1, (batch_size,))
    
#     transformer_encoder = StreamlineEncoder(
#         input_size=input_size,
#         d_model=cfg.d_model,
#         nhead=cfg.nhead,
#         num_layers=cfg.num_layers,
#         num_classes=num_classes
#     )
    
#     output = transformer_encoder(x, lengths=lengths)
#     print(f"  Input shape: {x.shape}")
#     print(f"  Output shape: {output.shape}")
#     print(f"  Parameters: {sum(p.numel() for p in transformer_encoder.parameters()):,}")