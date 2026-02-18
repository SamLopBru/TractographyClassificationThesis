import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .positional_encodings import SinusoidalPositionalEncoding, RotaryPositionalEncoding



class RoPETransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with Rotary Positional Encoding.
    
    Implements the same pre-norm architecture as
    nn.TransformerEncoderLayer(norm_first=True), but applies RoPE
    to Q and K inside the attention computation.
    
    Architecture: LayerNorm -> MHA(RoPE) -> Residual -> LayerNorm -> FFN -> Residual
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # RoPE
        self.rope = RotaryPositionalEncoding(self.d_head, max_len)
        
        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for attention
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass with RoPE attention.
        
        Args:
            src: (batch_size, seq_len, d_model)
            src_key_padding_mask: (batch_size, seq_len) True = padding
        
        Returns:
            (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = src.shape
        
        # Pre-norm self-attention with RoPE
        x = self.norm1(src)
        
        # Project Q, K, V and reshape for multi-head
        # (batch, seq, d_model) -> (batch, nhead, seq, d_head)
        q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2)
        
        # Apply RoPE to Q and K
        q = self.rope(q)
        k = self.rope(k)
        
        # Build attention mask from padding mask
        attn_mask = None
        if src_key_padding_mask is not None:
            # (batch, seq) -> (batch, 1, 1, seq) for broadcasting
            attn_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.to(dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(attn_mask.bool(), float('-inf'))
        
        # Scaled dot-product attention (uses Flash Attention when available)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        
        # Reshape back: (batch, nhead, seq, d_head) -> (batch, seq, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)
        
        # Residual connection
        x = src + self.attn_dropout(attn_out)
        
        # Pre-norm feedforward + residual
        x = x + self.ff(self.norm2(x))
        
        return x


class TransformerEncoder(nn.Module):
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
        pooling: str = 'cls',  # 'cls', 'mean', or 'max'
        pos_encoding: str = 'absolute'  # 'absolute' or 'rope'
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
        self.pos_encoding = pos_encoding
        
        # Input projection: (batch, seq, 5) -> (batch, seq, d_model)
        self.input_projection = nn.Linear(input_size, d_model)
        
        # CLS token for classification (learnable)
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        if pos_encoding == 'rope':
            # RoPE: positional info is injected inside each attention layer
            self.pos_encoder = None
            rope_layers = nn.ModuleList([
                RoPETransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    max_len=max_len
                )
                for _ in range(num_layers)
            ])
            self.transformer_encoder = rope_layers
        else:
            # Absolute sinusoidal positional encoding (default)
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len, dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True  # Pre-norm for better training stability
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, 
                num_layers=num_layers,
                enable_nested_tensor=True
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def get_embeddings(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor = None,
        padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Extract pooled embeddings before the classifier head.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, 5)
            lengths: Optional tensor of actual sequence lengths (batch_size,)
            padding_mask: Optional boolean mask where True indicates padding positions (batch_size, seq_len)
        
        Returns:
            Pooled embeddings of shape (batch_size, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Create padding mask from lengths if not provided
        if padding_mask is None and lengths is not None:
            indices = torch.arange(seq_len, device=x.device).unsqueeze(0)
            padding_mask = indices >= lengths.unsqueeze(1)
        
        # Project input to model dimension
        x = self.input_projection(x)  # (batch, seq, d_model)
        
        # Add CLS token if using cls pooling
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            if padding_mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        
        if self.pos_encoding == 'rope':
            # RoPE: no absolute PE, pass through custom RoPE layers
            for layer in self.transformer_encoder:
                x = layer(x, src_key_padding_mask=padding_mask)
        else:
            # Absolute PE
            x = x.transpose(0, 1)
            x = self.pos_encoder(x)
            x = x.transpose(0, 1)
            x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Pool the sequence
        if self.pooling == 'cls':
            pooled = x[:, 0]
        elif self.pooling == 'mean':
            if padding_mask is not None:
                mask = ~padding_mask.unsqueeze(-1)
                x = x * mask
                pooled = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)
        elif self.pooling == 'max':
            if padding_mask is not None:
                x = x.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
            pooled = x.max(dim=1)[0].clamp(min=-1e9)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        return pooled
    
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
        pooled = self.get_embeddings(x, lengths, padding_mask)
        logits = self.classifier(pooled)
        return logits


class LSTMEncoder(nn.Module):
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
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.output_size),
            nn.Linear(self.output_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def get_embeddings(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor = None,
        padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Extract pooled embeddings before the classifier head.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            lengths: Optional tensor of actual sequence lengths (batch_size,)
            padding_mask: Optional boolean mask where True indicates padding (batch_size, seq_len)
        
        Returns:
            Pooled embeddings of shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.shape
        
        if padding_mask is None and lengths is not None:
            indices = torch.arange(seq_len, device=x.device).unsqueeze(0)
            padding_mask = indices >= lengths.unsqueeze(1)
        
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            if lengths is not None:
                lengths = lengths + 1
            if padding_mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        
        if self.pooling == 'last':
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                _, (hidden, _) = self.lstm(packed)
            else:
                _, (hidden, _) = self.lstm(x)
            
            if self.num_directions == 2:
                pooled = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                pooled = hidden[-1]
        else:
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                output, _ = self.lstm(packed)
                output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            else:
                output, _ = self.lstm(x)
            
            if self.pooling == 'cls':
                pooled = output[:, 0]
            elif self.pooling == 'mean':
                if padding_mask is not None:
                    mask = ~padding_mask.unsqueeze(-1)
                    output = output * mask
                    pooled = output.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    pooled = output.mean(dim=1)
            elif self.pooling == 'max':
                if padding_mask is not None:
                    output = output.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
                pooled = output.max(dim=1)[0].clamp(min=-1e9)
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        return pooled
    
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
        pooled = self.get_embeddings(x, lengths, padding_mask)
        logits = self.classifier(pooled)
        return logits


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.
    
    Projects encoder embeddings into a lower-dimensional, L2-normalized space
    where contrastive loss is computed. Only used during contrastive pre-training.
    
    Architecture: Linear → BatchNorm → ReLU → Linear → L2-normalize
    Reference: "Supervised Contrastive Learning" (Khosla et al., 2020)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embeddings of shape (batch_size, input_dim)
        
        Returns:
            L2-normalized projections of shape (batch_size, output_dim)
        """
        projected = self.net(x)
        return F.normalize(projected, dim=1)


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
        # Filter out LSTM-only kwargs
        valid_keys = {'input_size', 'd_model', 'nhead', 'num_layers', 'dim_feedforward',
                      'max_len', 'dropout', 'pooling', 'pos_encoding'}
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
        return StreamlineEncoder(num_classes=num_classes, **filtered)
    elif encoder_type == 'lstm':
        return LightweightStreamlineEncoder(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

