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
    train_dir: str = "sequences/trainset"
    val_dir: str = "sequences/validset"
    sampling_pct: float = 0.10          # Percentage of streamlines to index per tract
    epoch_sampling_pct: float = 0.05    # Percentage of indexed streamlines per epoch
    min_samples_per_class: int = 10     # Minimum samples per class per epoch
    full_sample_threshold: int = 1000   # If tract has fewer than this, take 100%
    max_streamlines_per_tract: Optional[int] = None
    
    # Model configuration
    encoder_type: str = "transformer"  # "transformer" or "lstm"
    input_size: int = 5
    d_model: int = 256          # Larger for better capacity
    nhead: int = 8              # 256 / 8 = 32 dim per head
    num_layers: int = 6         # Deeper representations
    dim_feedforward: int = 512  # 2x d_model ratio
    num_classes: int = 32
    dropout: float = 0.1
    pooling: str = "mean"       # "cls", "mean", or "max"
    
    # Training configuration
    epochs: int = 20            # More epochs for convergence
    batch_size: int = 2048      # Safe for 16GB with d_model=256
    lr: float = 1e-4            # Lower LR for stability after warmup
    weight_decay: float = 1e-4  # More regularization
    accumulation_steps: int = 1
    patience: int = 5           # Early Stop patience
    use_amp: bool = True        # Essential for memory
    warmup_epochs: int = 5      # Linear warmup epochs
    plateau_patience: int = 3   # Epochs before LR reduction on plateau
    plateau_factor: float = 0.5 # LR reduction factor on plateau
    
    # System configuration
    num_workers: int = 8        
    save_dir: str = "checkpoints"
    log_interval: int = 50      # Less frequent logging
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.encoder_type in ["transformer", "lstm"], \
            f"encoder_type must be 'transformer' or 'lstm', got {self.encoder_type}"
        assert self.pooling in ["cls", "mean", "max"], \
            f"pooling must be 'cls', 'mean', or 'max', got {self.pooling}"
        assert 0 < self.sampling_pct <= 1, \
            f"sampling_pct must be between 0 and 1, got {self.sampling_pct}"
        assert 0 < self.epoch_sampling_pct <= 1, \
            f"epoch_sampling_pct must be between 0 and 1, got {self.epoch_sampling_pct}"


# Default configuration instance
DEFAULT_CONFIG = TrainConfig()
