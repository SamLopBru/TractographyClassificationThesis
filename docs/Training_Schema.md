# Training Schema — Streamline Bundle Classifier

This document describes the end-to-end training pipeline for classifying white-matter streamlines into anatomical bundles using sequential deep learning models.

---

## 1. Overview

The system classifies individual streamlines (variable-length sequences of 3D points encoded in spherical coordinates) into one of 32 anatomical bundles. Two encoder architectures are supported:

| Architecture | Class | Key Properties |
|---|---|---|
| **Transformer** | `TransformerEncoder` | Self-attention, positional encoding, pre-norm |
| **LSTM** | `LSTMEncoder` | Bidirectional, packed sequences |

The pipeline is implemented across four modules:

| Module | Role |
|---|---|
| `config.py` | Default hyperparameters (`TrainConfig` dataclass) |
| `dataloader.py` | Dataset indexing, stratified sampling, collation |
| `encoder.py` | Model architectures |
| `train.py` | Training loop, validation, logging, checkpointing |

---

## 2. Data Pipeline

### 2.1 Input Format

Streamlines are stored in HDF5 files, each containing groups `tract_XX` with:
- `streamlines`: array of shape `(n_streamlines, max_len, 5)` — 5 features per point: `(r_norm, θ_cos, θ_sin, φ_cos, φ_sin)`
- `lengths`: actual length of each streamline
- `tract_id`: class label (0–31)

### 2.2 Two-Level Sampling Strategy

Training uses **two levels of subsampling** for efficiency on large datasets:

```
Level 1: Dataset Indexing (sampling_pct = 10%)
    Full HDF5 data ──► Index 10% of streamlines per tract
    (done once at startup)

Level 2: Epoch Sampling (epoch_sampling_pct = 5%)
    Indexed streamlines ──► Sample 5% per class per epoch
    (different random subset each epoch)
```

- **Level 1** (`StreamlineDataset._build_index`): Indexes a fixed percentage of streamlines from each tract across all HDF5 files. Small tracts below `full_sample_threshold` (1000) are indexed at 100%.
- **Level 2** (`StratifiedEpochSampler`): Each epoch samples a different proportional subset from each class, ensuring class balance is preserved.

### 2.3 Validation Sampling

Validation uses a separate `StratifiedEpochSampler` with a higher sampling rate (`val_sampling_pct = 10%`) for more stable metric estimates. The validation sampler is deterministic (`shuffle=False`) within each epoch but varies across epochs.

### 2.4 Collation

`streamline_collate_fn` pads variable-length streamlines to the maximum length in each batch using `pad_sequence`, returning:
- `padded_streamlines`: `(batch_size, max_seq_len, 5)`
- `lengths`: `(batch_size,)` — actual lengths for masking
- `labels`: `(batch_size,)` — tract IDs

### 2.5 DataLoader Configuration

```python
DataLoader(
    batch_size=1024,      # Number of streamlines per batch
    num_workers=8,        # Parallel data loading processes
    pin_memory=True,      # Faster CPU→GPU transfers
    prefetch_factor=2,    # Batches prefetched per worker
    persistent_workers=False  # Free RAM between epochs
)
```

---

## 3. Model Architectures

### 3.1 Transformer Encoder (`StreamlineEncoder`)

```
Input (batch, seq, 5)
  │
  ├─ Linear(5 → d_model)            # Input projection
  │
  ├─ [CLS token prepended]          # If pooling='cls'
  │
  ├─ Sinusoidal Positional Encoding  # + dropout
  │
  ├─ TransformerEncoder              # num_layers × EncoderLayer
  │   └─ EncoderLayer:
  │       ├─ LayerNorm (pre-norm)
  │       ├─ MultiHeadAttention (nhead=8)
  │       ├─ LayerNorm (pre-norm)
  │       └─ FFN (d_model → dim_feedforward → d_model)
  │
  ├─ Pooling: cls | mean | max
  │
  └─ Classifier Head:
      LayerNorm → Linear(d_model → d_model) → GELU → Dropout → Linear(d_model → 32)
```

**Key design choices:**
- **Pre-norm** (`norm_first=True`): LayerNorm before attention/FFN for more stable training
- **Xavier initialization** for all 2D+ weight matrices
- **Padding mask** constructed from `lengths` to ignore padded positions in attention
- **CLS token**: Learnable `(1, 1, d_model)` parameter prepended to the sequence; the CLS position output is used for classification
- **`enable_nested_tensor=True`**: PyTorch optimization for variable-length sequences

### 3.2 LSTM Encoder (`LightweightStreamlineEncoder`)

```
Input (batch, seq, 5)
  │
  ├─ [CLS token prepended]          # If pooling='cls' (in input space)
  │
  ├─ pack_padded_sequence            # For efficient variable-length processing
  │
  ├─ Bidirectional LSTM              # num_layers deep, hidden_size per direction
  │
  ├─ Pooling: last | cls | mean | max
  │
  └─ Classifier Head:
      LayerNorm → Linear(hidden×2 → hidden) → GELU → Dropout → Linear(hidden → 32)
```

**Key design choices:**
- **Bidirectional** by default: output dimension = `hidden_size × 2`
- **Packed sequences**: Uses `pack_padded_sequence`/`pad_packed_sequence` for efficient computation on variable-length inputs
- **`last` pooling**: Concatenates final forward and backward hidden states
- **Inter-layer dropout** only when `num_layers > 1`
- **CLS token** operates in **input space** (5D), unlike the Transformer where it's in model space (d_model-D)

### 3.3 Default Architecture Parameters

| Parameter | Default | Description |
|---|---|---|
| `d_model` | 256 | Model/hidden dimension |
| `nhead` | 8 | Attention heads (32 dim/head) |
| `num_layers` | 6 | Encoder depth |
| `dim_feedforward` | 512 | FFN intermediate dimension (2× d_model) |
| `dropout` | 0.1 | Dropout rate |
| `pooling` | `"mean"` | Sequence pooling strategy |

---

## 4. Training Loop

### 4.1 Loss Function

```python
CrossEntropyLoss(label_smoothing=0.1)
```

Label smoothing distributes 10% of the probability mass uniformly across all classes, preventing the model from becoming overconfident and improving generalization.

### 4.2 Optimizer

**AdamW** with architecture-specific weight decay:

| Architecture | Weight Decay |
|---|---|
| Transformer | `1e-4` |
| LSTM | `1e-5` (lower — LSTMs are more sensitive) |

**Learning rate scaling**: LR is scaled with batch size using the square-root rule:

```
lr = base_lr × √(effective_batch_size / 256)
```

Where `effective_batch_size = batch_size × accumulation_steps`. Default: `base_lr=1e-4`, `batch_size=1024`, `accumulation_steps=2` → `lr ≈ 2.83e-4`.

### 4.3 Learning Rate Schedule

Two schedulers work together:

#### Primary: Warmup + Cosine Annealing (per step)
```
Steps 0–500:        Linear warmup from 10% → 100% of target LR
Steps 500–end:      Cosine decay from 100% → 1% of target LR
```

Stepped **every optimizer step** (every `accumulation_steps` batches) inside `train_epoch()`.

#### Backup: ReduceLROnPlateau (per epoch)
```
Monitors:           Validation loss
Patience:           3 epochs without improvement
Reduction factor:   0.5×
Minimum LR:         1e-7
```

Stepped at the end of each validation epoch.

### 4.4 Mixed Precision Training (AMP)

- Uses `torch.amp.autocast(dtype=torch.float16)` for forward pass
- `GradScaler` handles loss scaling and gradient unscaling
- Inf/NaN gradient detection: batches with invalid gradients are skipped with a warning
- **TensorFloat-32** enabled for matmul on Ampere+ GPUs

### 4.5 Gradient Accumulation

Effective batch size is multiplied by `accumulation_steps` (default: 2):
- Loss is divided by `accumulation_steps` before backward
- Optimizer steps only every `accumulation_steps` batches
- Handles remaining gradients at the end of each epoch

### 4.6 Gradient Clipping

```python
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Applied after unscaling gradients (with AMP) or directly (without AMP). Prevents gradient explosion especially during early training.

### 4.7 Exponential Moving Average (EMA)

After each optimizer step, EMA parameters are updated:

```python
ema_param = ema_decay × ema_param + (1 - ema_decay) × model_param
```

With `ema_decay=0.999`. The EMA model:
- Is validated alongside the regular model (logged as `[EMA]`)
- Is saved separately as `best_model_ema.pt`
- Provides smoother, often better final weights

### 4.8 Validation Scheduling

To save time during warmup:
- **Warmup period**: Validate every `validate_every × 2` epochs
- **After warmup**: Validate every `validate_every` epochs (default: 2)
- **Final epoch**: Always validated

When validation is skipped, the previous validation metrics are carried forward.

### 4.9 Early Stopping

- **Metric**: Macro F1 score (treats all 32 classes equally)
- **Patience**: 5 epochs without improving best F1
- **Best model saved**: Based on highest validation Macro F1

### 4.10 Checkpointing

Two checkpoints are maintained:

| File | Contents | When Saved |
|---|---|---|
| `best_model.pt` | Model + EMA + optimizer + metrics | On new best F1 |
| `latest_checkpoint.pt` | Full state (schedulers, scaler, history) | Every epoch |

Training can be resumed from `latest_checkpoint.pt` using `--resume`.

---

## 5. Performance Optimizations

| Optimization | Description |
|---|---|
| `torch.compile(dynamic=True)` | JIT compilation for variable-length sequences |
| `cudnn.benchmark = True` | Auto-tune convolution algorithms |
| `TF32 matmul precision` | Faster matrix multiplications on Ampere+ GPUs |
| `non_blocking=True` | Async CPU→GPU transfers with pinned memory |
| GPU-side metric accumulation | Loss/accuracy tracked as CUDA tensors to avoid sync |
| File handle caching | HDF5 handles cached per-worker to reduce I/O |
| NumPy structured arrays | Dataset index uses ~90% less RAM than Python dicts |

---

## 6. Monitoring & Logging

### TensorBoard

Logged every epoch to `runs/YYYYMMDD_HHMMSS/`:
- `Loss/train`, `Loss/val`
- `Accuracy/train`, `Accuracy/val`
- `Val_Macro_F1`
- `Learning_Rate`
- `Gradient_Norm`

### Experiments CSV

Each completed run appends a row to `experiments.csv` with:
- Timestamp, experiment name, encoder type
- Best val F1, best val accuracy, best epoch
- All hyperparameters and parameter count

### Console Output

Per-batch progress (every `log_interval=50` batches) and per-epoch summaries with train/val metrics and timing.

---

## 7. Typical Training Command

```bash
# Default training (Transformer, mean pooling, d_model=256, 6 layers)
uv run src/train.py

# LSTM with CLS pooling
uv run src/train.py --encoder_type lstm --pooling cls --d_model 256 --num_layers 4

# Transformer CLS with more data per epoch
uv run src/train.py --pooling cls --epoch_sampling_pct 0.15 --val_sampling_pct 0.2

# Resume from checkpoint
uv run src/train.py --resume checkpoints/latest_checkpoint.pt
```

---

## 8. Training Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     STARTUP                             │
│  Load HDF5 files → Build index (10% per tract)         │
│  Create model → torch.compile → Scale LR               │
├─────────────────────────────────────────────────────────┤
│                  EPOCH LOOP                             │
│                                                         │
│  1. StratifiedEpochSampler: new 5% subset per class    │
│  2. train_epoch():                                      │
│     ├─ Forward (AMP) → Loss (label smoothing)          │
│     ├─ Backward → Gradient accumulation (2 steps)      │
│     ├─ Clip gradients (max_norm=1.0)                   │
│     ├─ Optimizer step → LR scheduler step              │
│     └─ EMA update                                       │
│  3. Validation (if scheduled):                          │
│     ├─ Regular model metrics                            │
│     ├─ EMA model metrics                                │
│     └─ Plateau scheduler step                           │
│  4. Checkpoint if best F1 → Early stop check           │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                    OUTPUTS                              │
│  checkpoints/best_model.pt                              │
│  checkpoints/best_model_ema.pt                          │
│  checkpoints/history.json                               │
│  experiments.csv (appended)                             │
│  runs/<timestamp>/ (TensorBoard)                        │
└─────────────────────────────────────────────────────────┘
```
