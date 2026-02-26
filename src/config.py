"""
Configuration class for training the Streamline Bundle Classifier.

This module provides default configuration values that are used when 
command-line arguments are not provided.

Optimized for RTX 5060 Ti (16GB VRAM).
"""

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class TrainConfig:
    """Configuration for training the streamline encoder model."""
    
    # Data configuration
    train_dir: str = "sequences/trainset"
    val_dir: str = "sequences/validset"
    sampling_pct: float = 0.10          # Percentage of streamlines to index per tract
    epoch_sampling_pct: float = 0.05    # Percentage of indexed streamlines per epoch
    val_sampling_pct: float = 0.10      # Percentage of validation streamlines to use (higher = more stable metrics)
    min_samples_per_class: int = 10     # Minimum samples per class per epoch
    full_sample_threshold: Union[int, float] = 1000   # If tract has fewer than this, take 100%
    max_streamlines_per_tract: Optional[int] = None
    
    # Model configuration
    encoder_type: str = "transformer"  # "transformer" or "lstm"
    input_size: int = 5
    d_model: int = 256          
    nhead: int = 8              # 256 / 8 = 32 dim per head
    num_layers: int = 6         
    dim_feedforward: int = 512  # 2x d_model ratio
    num_classes: int = 32
    dropout: float = 0.1
    pooling: str = "cls"       # "cls", "mean", or "max"
    pos_encoding: str = "absolute"  # "absolute" or "rope"
    norm_layer: str = "layernorm"  # "layernorm" or "rmsnorm"
    
    # Training configuration
    epochs: int = 20            
    batch_size: int = 1024      
    base_lr: float = 1e-4       # Base LR (will be scaled by effective batch size)
    lr_scale_with_batch: bool = True  # Scale LR with sqrt(batch_size/256)
    weight_decay_transformer: float = 1e-4  # Weight decay for Transformer
    weight_decay_lstm: float = 1e-5         # Weight decay for LSTM (lower)
    accumulation_steps: int = 2
    patience: int = 5           # Early Stop patience
    label_smoothing: float = 0.1  # Label smoothing for CrossEntropyLoss (0.0 = disabled)
    loss_type: str = "cross_entropy"  # "cross_entropy", "focal", or "supcon_hybrid"
    focal_gamma: float = 2.0  # Focal Loss gamma (higher = more focus on hard examples)
    
    # Supervised Contrastive Loss (hybrid mode)
    supcon_weight: float = 0.1        # Lambda weight for SupCon term in hybrid loss
    supcon_temperature: float = 0.07  # SupCon temperature (lower = sharper)
    projection_dim: int = 128         # Projection head output dimension
    use_amp: bool = True        # Activate Automatic Mixed Precision
    warmup_steps: int = 1500    # Warmup by steps 
    scheduler: str = "cosine_plateau"  # "cosine_plateau", "cosine_only" or "cosine_restarts"
    plateau_patience: int = 5   # Epochs before LR reduction on plateau
    plateau_factor: float = 0.5 # LR reduction factor on plateau
    max_grad_norm: float = 1.0  # Maximum gradient norm for clipping
    use_ema: bool = True        # Use Exponential Moving Average
    ema_decay: float = 0.999    # EMA decay factor
    use_swa: bool = False       # Use Stochastic Weight Averaging (better generalization)
    swa_start_epoch: int = 10   # Start averaging from this epoch
    validate_every: int = 2     # Validate every N epochs
    
    # Cosine Annealing with Warm Restarts configuration
    T_0: int = 10               # Number of epochs for the first restart
    T_mult: int = 2             # Factor to increase the cycle length after each restart
    
    # System configuration
    num_workers: int = 8        
    save_dir: str = "checkpoints"
    log_interval: int = 50    
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.encoder_type in ["transformer", "lstm"], \
            f"encoder_type must be 'transformer' or 'lstm', got {self.encoder_type}"
        assert self.pooling in ["cls", "mean", "max"], \
            f"pooling must be 'cls', 'mean', or 'max', got {self.pooling}"
        assert self.pos_encoding in ["absolute", "rope"], \
            f"pos_encoding must be 'absolute' or 'rope', got {self.pos_encoding}"
        assert self.norm_layer in ["layernorm", "rmsnorm"], \
            f"norm_layer must be 'layernorm' or 'rmsnorm', got {self.norm_layer}"
        assert self.scheduler in ["cosine_plateau", "cosine_only", "cosine_restarts"], \
            f"scheduler must be 'cosine_plateau', 'cosine_only' or 'cosine_restarts', got {self.scheduler}"
        assert self.loss_type in ["cross_entropy", "focal", "supcon_hybrid"], \
            f"loss_type must be 'cross_entropy', 'focal', or 'supcon_hybrid', got {self.loss_type}"
        assert 0 < self.sampling_pct <= 1, \
            f"sampling_pct must be between 0 and 1, got {self.sampling_pct}"
        assert 0 < self.epoch_sampling_pct <= 1, \
            f"epoch_sampling_pct must be between 0 and 1, got {self.epoch_sampling_pct}"


# Default configuration instance
DEFAULT_CONFIG = TrainConfig()