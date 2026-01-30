"""
Configuration class for training the Streamline Bundle Classifier.

This module provides default configuration values that are used when 
command-line arguments are not provided.

Optimized for RTX 5060 Ti (16GB VRAM).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """Configuration for training the streamline encoder model."""
    
    # Data configuration
    data_dir: str = "sequences/trainset"
    val_split: float = 0.2
    sampling_pct: float = 0.1
    max_streamlines_per_tract: Optional[int] = None
    
    # Model configuration
    encoder_type: str = "transformer"  # "transformer" or "lstm"
    input_size: int = 5
    d_model: int = 256          # Larger for better capacity
    nhead: int = 8              # 256 / 8 = 32 dim per head
    num_layers: int = 6         # Deeper representations
    dim_feedforward: int = 1024 # 4x d_model ratio
    num_classes: int = 32
    dropout: float = 0.1
    pooling: str = "mean"       # "cls", "mean", or "max"
    
    # Training configuration
    epochs: int = 100           # More epochs for convergence
    batch_size: int = 512       # Safe for 16GB with d_model=256
    lr: float = 3e-4            # Higher LR for larger batch
    weight_decay: float = 1e-4  # More regularization
    accumulation_steps: int = 1
    patience: int = 15          # More patience
    use_amp: bool = True        # Essential for memory
    
    # System configuration
    num_workers: int = 8        # Better parallelism
    save_dir: str = "checkpoints"
    log_interval: int = 50      # Less frequent logging
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.encoder_type in ["transformer", "lstm"], \
            f"encoder_type must be 'transformer' or 'lstm', got {self.encoder_type}"
        assert self.pooling in ["cls", "mean", "max"], \
            f"pooling must be 'cls', 'mean', or 'max', got {self.pooling}"
        assert 0 < self.val_split < 1, \
            f"val_split must be between 0 and 1, got {self.val_split}"
        assert 0 < self.sampling_pct <= 1, \
            f"sampling_pct must be between 0 and 1, got {self.sampling_pct}"


# Default configuration instance
DEFAULT_CONFIG = TrainConfig()
