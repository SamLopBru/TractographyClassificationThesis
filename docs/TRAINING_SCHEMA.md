# Training Schema Documentation

This document describes the training pipeline for the Streamline Bundle Classifier model.

---

## Overview

The training pipeline is designed for **flexibility**, **efficiency**, and **robustness** across different model architectures (Transformer and LSTM). It includes modern best practices for deep learning training.

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│  1. Data Loading (Stratified Sampling)                      │
│  2. LR Warmup (Step-based) + Cosine Annealing               │
│  3. Mixed Precision (FP16) Training                         │
│  4. EMA Weight Averaging                                    │
│  5. Validation with Macro F1                                │
│  6. Early Stopping + Checkpointing                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Loading

### Two-Level Sampling Strategy

| Level | Description | Default |
|-------|-------------|---------|
| **Indexing** | Percentage of streamlines indexed from HDF5 files | 10% |
| **Epoch Sampling** | Percentage of indexed samples used per epoch | 5% |

This allows training on large datasets without loading everything into memory.

### Stratified Sampling

Both training and validation use `StratifiedEpochSampler`:
- Guarantees proportional representation of all classes each epoch
- Ensures minority classes are never missed
- Essential for reliable **Macro F1** metrics

---

## Learning Rate Schedule

### Step-based Warmup + Cosine Annealing

```
LR
 ▲
 │    ┌──────────────────╮
 │   /                    ╲
 │  /                      ╲
 │ /                        ╲
 │/                          ╲_____
 └──────────────────────────────────► Steps
   │← Warmup →│←─ Cosine Annealing ─→│
     (500 steps)
```

**Formula:**
- **Warmup phase** (steps < 500): `LR = base_lr × (0.1 + 0.9 × step/500)`
- **Cosine phase** (steps ≥ 500): `LR = base_lr × (0.01 + 0.99 × 0.5 × (1 + cos(π × progress)))`

### LR Scaling with Batch Size

Learning rate is automatically scaled based on batch size for better convergence:

```python
lr = base_lr × sqrt(batch_size / 256)
```

| Batch Size | Scaling Factor | Effective LR (base=1e-4) |
|------------|----------------|--------------------------|
| 256 | 1.0× | 1.0e-4 |
| 512 | 1.41× | 1.41e-4 |
| 1024 | 2.0× | 2.0e-4 |
| 2048 | 2.83× | 2.83e-4 |

Disable with `--no_lr_scaling`.

### Plateau Backup Scheduler

If validation loss plateaus for 3 epochs, LR is reduced by 50%:
- `patience=3`, `factor=0.5`, `min_lr=1e-7`

---

## Architecture-Specific Weight Decay

Different architectures benefit from different regularization:

| Architecture | Weight Decay | Reason |
|--------------|--------------|--------|
| **Transformer** | `1e-4` | Higher regularization for attention layers |
| **LSTM** | `1e-5` | Recurrent networks need less regularization |

Automatically selected based on `--encoder_type`.

---

## Mixed Precision Training (AMP)

Uses **Automatic Mixed Precision** with FP16:
- Reduces memory usage by ~50%
- Speeds up training on modern GPUs (RTX 30xx/40xx/50xx)
- Uses `GradScaler` for stable gradients

Disable with `--no_amp`.

---

## Exponential Moving Average (EMA)

EMA maintains a smoothed version of model weights:

```python
ema_weights = decay × ema_weights + (1 - decay) × model_weights
```

| Parameter | Value |
|-----------|-------|
| Decay | 0.999 |
| Update frequency | Every optimizer step |

**Benefits:**
- Reduces noise in final model
- Often improves accuracy by 0.5-1%
- Both regular and EMA models are validated (EMA metrics shown separately)

**Saved checkpoints:**
- `best_model.pt` - Contains both regular and EMA weights
- `best_model_ema.pt` - EMA weights only (for inference)

Disable with `--no_ema`.

---

## Validation Scheduling

Validation is less frequent during warmup phase to save time:

| Training Phase | Validation Frequency |
|----------------|----------------------|
| During warmup | Every 2 epochs |
| After warmup | Every 1 epoch (configurable) |
| Final epoch | Always |

Configure with `--validate_every N`.

---

## Loss Function

**CrossEntropyLoss with Label Smoothing:**
- `label_smoothing=0.1`
- Prevents overconfident predictions
- Improves generalization

---

## Early Stopping

Training stops if Macro F1 doesn't improve for `patience` epochs:
- Default patience: 5 epochs
- Tracks best model based on **Macro F1** (not accuracy)
- Important for imbalanced datasets

---

## Checkpointing

### Saved Files

| File | Contents |
|------|----------|
| `best_model.pt` | Best model (regular + EMA), optimizer, metrics |
| `best_model_ema.pt` | EMA model only |
| `latest_checkpoint.pt` | Full training state for resume |
| `history.json` | Training history |

### Resume Training

```bash
python src/train.py --resume checkpoints/latest_checkpoint.pt
```

Restores: model, EMA, optimizer, schedulers, scaler, epoch, history.

---

## GPU Optimizations

| Optimization | Description |
|--------------|-------------|
| `cudnn.benchmark=True` | Auto-tune convolution algorithms |
| `TF32 precision` | Faster matmul on Ampere+ GPUs |
| `non_blocking=True` | Async CPU→GPU transfers |
| `pin_memory=True` | Faster data loading |
| `torch.compile` | JIT compilation for faster execution |

---

## Command-Line Arguments

### Data Arguments
```
--train_dir         Training HDF5 directory
--val_dir           Validation HDF5 directory
--sampling_pct      % of streamlines to index (default: 0.10)
--epoch_sampling_pct % of indexed samples per epoch (default: 0.05)
```

### Model Arguments
```
--encoder_type      transformer | lstm
--d_model           Model dimension (default: 256)
--num_layers        Number of layers (default: 6)
--nhead             Attention heads (default: 8)
--pooling           cls | mean | max
```

### Training Arguments
```
--epochs            Number of epochs (default: 20)
--batch_size        Batch size (default: 2048)
--base_lr           Base learning rate (default: 1e-4)
--no_lr_scaling     Disable LR scaling with batch size
--no_amp            Disable mixed precision
--no_ema            Disable EMA
--validate_every    Validation frequency (default: 1)
--patience          Early stopping patience (default: 5)
--resume            Resume from checkpoint
```

---

## Example Usage

```bash
# Train with default settings (Transformer)
python src/train.py

# Train LSTM with custom batch size
python src/train.py --encoder_type lstm --batch_size 1024

# Resume training
python src/train.py --resume checkpoints/latest_checkpoint.pt

# Disable EMA for faster training
python src/train.py --no_ema

# Custom LR without scaling
python src/train.py --base_lr 5e-5 --no_lr_scaling
```

---

## TensorBoard Monitoring

```bash
tensorboard --logdir=runs
```

**Tracked metrics:**
- Train/Val Loss
- Train/Val Accuracy
- Validation Macro F1
- Learning Rate
- Gradient Norm
