import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
import os
import sys
import time
import argparse
from typing import Dict, List, Tuple, Optional
import json
import math
import gc
import copy
import numpy as np
import csv
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

# Enable cuDNN benchmarking for faster training (finds optimal algorithms for your hardware)
torch.backends.cudnn.benchmark = True

# Enable TensorFloat-32 for faster matmul on Ampere+ GPUs (RTX 30xx, 40xx, 50xx)
torch.set_float32_matmul_precision('high')

# Fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.encoder import StreamlineEncoder, LightweightStreamlineEncoder
from src.dataloader import StreamlineDataset, StratifiedEpochSampler, EpochSubsetSampler, streamline_collate_fn
from src.config import TrainConfig, DEFAULT_CONFIG


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: GradScaler = None,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    log_interval: int = 10,
    scheduler: optim.lr_scheduler._LRScheduler = None,
    ema_model: nn.Module = None,
    ema_decay: float = 0.999
) -> Dict[str, float]:
    """Train for one epoch with optional mixed precision, gradient accumulation, and EMA."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_grad_norm = 0.0
    grad_norm_count = 0
    
    start_time = time.time()
    optimizer.zero_grad()
    
    for batch_idx, (streamlines, lengths, labels) in enumerate(dataloader):
        # Use non_blocking=True with pin_memory for async H2D transfers
        streamlines = streamlines.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass with optional mixed precision
        if use_amp and scaler is not None:
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(streamlines, lengths=lengths)
                loss = criterion(logits, labels) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Check for inf/nan gradients and skip if found
                if torch.isfinite(grad_norm):
                    total_grad_norm += grad_norm.item()
                    grad_norm_count += 1
                    scaler.step(optimizer)
                else:
                    print(f"  ⚠ Warning: Skipping batch {batch_idx+1} due to inf/nan gradients")
                
                scaler.update()
                optimizer.zero_grad()
                
                # Step-based LR scheduler update
                if scheduler is not None:
                    scheduler.step()
                
                # EMA update
                if ema_model is not None:
                    with torch.no_grad():
                        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
                            ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1 - ema_decay)
        else:
            logits = model(streamlines, lengths=lengths)
            loss = criterion(logits, labels) / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Check for inf/nan gradients and skip if found
                if torch.isfinite(grad_norm):
                    total_grad_norm += grad_norm.item()
                    grad_norm_count += 1
                    optimizer.step()
                else:
                    print(f"  ⚠ Warning: Skipping batch {batch_idx+1} due to inf/nan gradients")
                
                optimizer.zero_grad()
                
                # Step-based LR scheduler update
                if scheduler is not None:
                    scheduler.step()
                
                # EMA update
                if ema_model is not None:
                    with torch.no_grad():
                        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
                            ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1 - ema_decay)
        
        # Metrics (multiply back for logging)
        total_loss += loss.item() * accumulation_steps * labels.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / total_samples
            accuracy = 100.0 * total_correct / total_samples
            elapsed = time.time() - start_time
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | "
                  f"Time: {elapsed:.1f}s")
    
    # Handle any remaining gradients
    if (batch_idx + 1) % accumulation_steps != 0:
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': 100.0 * total_correct / total_samples,
        'grad_norm': total_grad_norm / max(1, grad_norm_count)
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True
) -> Dict[str, float]:
    """Validate the model and compute metrics including Macro F1."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    all_labels = []
    all_preds = []
    
    for streamlines, lengths, labels in dataloader:
        # Use non_blocking=True with pin_memory for async H2D transfers
        streamlines = streamlines.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(streamlines, lengths=lengths)
                loss = criterion(logits, labels)
        else:
            logits = model(streamlines, lengths=lengths)
            loss = criterion(logits, labels)
        
        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        # Collect for F1 computation
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predictions.cpu().numpy())
    
    # Compute Macro F1 (treats all classes equally)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': 100.0 * total_correct / total_samples,
        'macro_f1': macro_f1
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    save_dir: str = "checkpoints",
    patience: int = 10,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    train_sampler: StratifiedEpochSampler = None,
    val_sampler: 'StratifiedEpochSampler' = None,
    warmup_steps: int = 500,
    plateau_patience: int = 3,
    plateau_factor: float = 0.5,
    train_dataset: 'StreamlineDataset' = None,
    num_classes: int = 32,
    label_smoothing: float = 0.1,
    resume_checkpoint: str = None,
    use_ema: bool = True,
    ema_decay: float = 0.999,
    validate_every: int = 1
) -> Dict[str, List[float]]:
    """
    Full training loop with validation, early stopping, warmup, and mixed precision.
    
    Features:
    - LR warmup by steps (not epochs) for consistent warmup across batch sizes
    - Cosine annealing after warmup with ReduceLROnPlateau as backup
    - EMA (Exponential Moving Average) for better final model quality
    - Validation scheduling (less frequent during early training)
    - Label smoothing for better generalization
    """
    model = model.to(device)
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision (FP16) training")
    
    # Loss function with label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    print(f"Using CrossEntropyLoss with label_smoothing={label_smoothing}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Calculate total steps for warmup scheduling
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    
    # Step-based warmup + cosine annealing scheduler
    def warmup_cosine_schedule_fn(current_step):
        if current_step < warmup_steps:
            # Linear warmup: start at 10% and increase to 100%
            return 0.1 + 0.9 * (current_step / warmup_steps)
        else:
            # Cosine annealing after warmup: decay from 100% to 1%
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    warmup_cosine_scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_schedule_fn)
    
    # Backup scheduler: ReduceLROnPlateau for when validation loss plateaus
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',  # Monitor validation loss
        factor=plateau_factor,
        patience=plateau_patience,
        min_lr=1e-7
    )
    
    print(f"Scheduler: {warmup_steps} steps warmup + cosine annealing (total {total_steps} steps)")
    print(f"Plateau LR reduction: factor={plateau_factor}, patience={plateau_patience}")
    
    # EMA model for better final quality
    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        print(f"Using EMA with decay={ema_decay}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'epoch_time': []  # Track time per epoch
    }
    
    best_val_f1 = 0.0
    patience_counter = 0
    start_epoch = 1
    
    # Resume from checkpoint if provided
    if resume_checkpoint is not None:
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        
        # Load model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler states if available
        if 'warmup_cosine_scheduler_state' in checkpoint:
            warmup_cosine_scheduler.load_state_dict(checkpoint['warmup_cosine_scheduler_state'])
        if 'plateau_scheduler_state' in checkpoint:
            plateau_scheduler.load_state_dict(checkpoint['plateau_scheduler_state'])
        
        # Load scaler state if using AMP
        if scaler is not None and checkpoint.get('scaler_state') is not None:
            scaler.load_state_dict(checkpoint['scaler_state'])
        
        # Restore training state
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint.get('history', history)
        best_val_f1 = checkpoint.get('best_val_f1', max(history['val_f1']) if history['val_f1'] else 0.0)
        patience_counter = checkpoint.get('patience_counter', 0)
        
        print(f"  Resumed at epoch {start_epoch}, best F1: {best_val_f1:.2f}%")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # TensorBoard setup
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=os.path.join('runs', run_name))
    print(f"TensorBoard logs: runs/{run_name}")
    
    # Log hyperparameters
    hparams = {
        'epochs': epochs,
        'lr': lr,
        'weight_decay': weight_decay,
        'warmup_steps': warmup_steps,
        'plateau_patience': plateau_patience,
        'plateau_factor': plateau_factor,
        'accumulation_steps': accumulation_steps,
        'use_ema': use_ema,
        'ema_decay': ema_decay,
        'validate_every': validate_every
    }
    writer.add_text('Hyperparameters', str(hparams), 0)
    
    # Calculate warmup epoch threshold for validation scheduling
    warmup_epoch_threshold = max(1, warmup_steps // steps_per_epoch)
    
    for epoch in range(start_epoch, epochs + 1):
        # Update sampler for new epoch (different random subset)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print('='*60)
        
        epoch_start_time = time.time()
        
        # Train (scheduler steps inside train_epoch, not here)
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler, use_amp=use_amp,
            accumulation_steps=accumulation_steps,
            scheduler=warmup_cosine_scheduler,
            ema_model=ema_model,
            ema_decay=ema_decay
        )
        print(f"\n  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        
        # Update validation sampler for each epoch
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        
        # Validation scheduling: validate less frequently during warmup
        current_validate_every = validate_every * 2 if epoch <= warmup_epoch_threshold else validate_every
        should_validate = (epoch % current_validate_every == 0) or (epoch == epochs)
        
        if should_validate:
            # Validate with regular model
            val_metrics = validate(model, val_loader, criterion, device, use_amp=use_amp)
            
            # If using EMA, also validate with EMA model and report
            if ema_model is not None:
                ema_val_metrics = validate(ema_model, val_loader, criterion, device, use_amp=use_amp)
                print(f"  [EMA] Val Loss: {ema_val_metrics['loss']:.4f} | Val Acc: {ema_val_metrics['accuracy']:.2f}% | Val F1: {ema_val_metrics['macro_f1']:.2f}%")
        else:
            # Use placeholder metrics when skipping validation
            val_metrics = {'loss': history['val_loss'][-1] if history['val_loss'] else 0, 
                          'accuracy': history['val_acc'][-1] if history['val_acc'] else 0,
                          'macro_f1': history['val_f1'][-1] if history['val_f1'] else 0}
            print(f"  [Skipping validation this epoch]")
        
        epoch_time = time.time() - epoch_start_time
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}% | Val F1: {val_metrics['macro_f1']:.2f}% | Time: {epoch_time:.1f}s")
        
        # Plateau scheduler uses val loss (only on validation epochs)
        if should_validate:
            plateau_scheduler.step(val_metrics['loss'])
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['macro_f1'])
        history['epoch_time'].append(epoch_time)
        
        # TensorBoard logging
        writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        writer.add_scalars('Accuracy', {
            'train': train_metrics['accuracy'],
            'val': val_metrics['accuracy']
        }, epoch)
        writer.add_scalar('Val_Macro_F1', val_metrics['macro_f1'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Gradient_Norm', train_metrics['grad_norm'], epoch)
        
        # Save best model (based on Macro F1 for better handling of imbalanced classes)
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict() if ema_model is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'val_macro_f1': best_val_f1,
                'history': history
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
            
            # Also save EMA model separately if available
            if ema_model is not None:
                torch.save({'model_state_dict': ema_model.state_dict()}, 
                          os.path.join(save_dir, 'best_model_ema.pt'))
            print(f"  ✓ Saved new best model (Val F1: {best_val_f1:.2f}%, Acc: {val_metrics['accuracy']:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping triggered after {epoch} epochs")
                break
        
        # Save latest checkpoint with all states needed for resume
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': ema_model.state_dict() if ema_model is not None else None,
            'optimizer_state_dict': optimizer.state_dict(),
            'warmup_cosine_scheduler_state': warmup_cosine_scheduler.state_dict(),
            'plateau_scheduler_state': plateau_scheduler.state_dict(),
            'scaler_state': scaler.state_dict() if scaler is not None else None,
            'best_val_f1': best_val_f1,
            'patience_counter': patience_counter,
            'history': history
        }, os.path.join(save_dir, 'latest_checkpoint.pt'))
        
        # Memory cleanup at end of each epoch
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Save training history
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best validation Macro F1: {best_val_f1:.2f}%")
    print(f"View TensorBoard: tensorboard --logdir=runs")
    print('='*60)
    
    return history


def log_experiment(
    experiment_name: str,
    encoder_type: str,
    history: Dict[str, List[float]],
    params: Dict,
    csv_path: str = "experiments.csv"
):
    """
    Log experiment results to a CSV file for comparison.
    
    Args:
        experiment_name: Name/description of this experiment
        encoder_type: 'transformer' or 'lstm'
        history: Training history dict with loss, accuracy, f1
        params: Dict of hyperparameters
        csv_path: Path to CSV file (created if doesn't exist)
    """
    # Get best metrics
    best_val_f1 = max(history['val_f1']) if history['val_f1'] else 0
    best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
    best_epoch = history['val_f1'].index(best_val_f1) + 1 if history['val_f1'] else 0
    final_train_loss = history['train_loss'][-1] if history['train_loss'] else 0
    final_val_loss = history['val_loss'][-1] if history['val_loss'] else 0
    mean_epoch_time = sum(history.get('epoch_time', [0])) / max(1, len(history.get('epoch_time', [1])))
    
    # Create row
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiment': experiment_name,
        'encoder_type': encoder_type,
        'best_val_f1': f"{best_val_f1:.2f}",
        'best_val_acc': f"{best_val_acc:.2f}",
        'best_epoch': best_epoch,
        'total_epochs': len(history['train_loss']),
        'mean_epoch_time': f"{mean_epoch_time:.1f}",
        'final_train_loss': f"{final_train_loss:.4f}",
        'final_val_loss': f"{final_val_loss:.4f}",
        'd_model': params.get('d_model', ''),
        'dim_feedforward': params.get('dim_feedforward', ''),
        'num_layers': params.get('num_layers', ''),
        'nhead': params.get('nhead', ''),
        'lr': params.get('lr', ''),
        'batch_size': params.get('batch_size', ''),
        'dropout': params.get('dropout', ''),
        'pooling': params.get('pooling', ''),
        'num_workers': params.get('num_workers', ''),
        'warmup_epochs': params.get('warmup_epochs', ''),
        'plateau_patience': params.get('plateau_patience', ''),
        'plateau_factor': params.get('plateau_factor', ''),
        'parameters': params.get('parameters', ''),
    }
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"\n📊 Experiment logged to {csv_path}")


def main():
    # Use DEFAULT_CONFIG values as defaults for argparse
    cfg = DEFAULT_CONFIG
    
    parser = argparse.ArgumentParser(description='Train Streamline Bundle Classifier')
    
    # Data arguments
    parser.add_argument('--train_dir', type=str, default=cfg.train_dir,
                        help='Directory containing training HDF5 files')
    parser.add_argument('--val_dir', type=str, default=cfg.val_dir,
                        help='Directory containing validation HDF5 files')
    parser.add_argument('--sampling_pct', type=float, default=cfg.sampling_pct,
                        help='Percentage of streamlines to index per tract')
    parser.add_argument('--epoch_sampling_pct', type=float, default=cfg.epoch_sampling_pct,
                        help='Percentage of indexed streamlines to use per epoch')
    parser.add_argument('--min_samples_per_class', type=int, default=cfg.min_samples_per_class,
                        help='Minimum samples per class per epoch')
    parser.add_argument('--full_sample_threshold', type=int, default=cfg.full_sample_threshold,
                        help='If tract has fewer streamlines than this, take all 100%')
    parser.add_argument('--max_streamlines_per_tract', type=int, default=cfg.max_streamlines_per_tract,
                        help='Max streamlines per tract (None = no limit)')
    
    # Model arguments
    parser.add_argument('--encoder_type', type=str, default=cfg.encoder_type,
                        choices=['transformer', 'lstm'], help='Type of encoder')
    parser.add_argument('--d_model', type=int, default=cfg.d_model,
                        help='Model dimension')
    parser.add_argument('--nhead', type=int, default=cfg.nhead,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=cfg.num_layers,
                        help='Number of layers')
    parser.add_argument('--dim_feedforward', type=int, default=cfg.dim_feedforward,
                        help='Feedforward dimension')
    parser.add_argument('--num_classes', type=int, default=cfg.num_classes,
                        help='Number of bundle classes')
    parser.add_argument('--dropout', type=float, default=cfg.dropout,
                        help='Dropout rate')
    parser.add_argument('--pooling', type=str, default=cfg.pooling,
                        choices=['cls', 'mean', 'max'], help='Pooling strategy')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=cfg.epochs,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size,
                        help='Batch size (streamlines)')
    parser.add_argument('--base_lr', type=float, default=cfg.base_lr,
                        help='Base learning rate (will be scaled with batch size if enabled)')
    parser.add_argument('--no_lr_scaling', action='store_true',
                        help='Disable LR scaling with batch size')
    parser.add_argument('--accumulation_steps', type=int, default=cfg.accumulation_steps,
                        help='Gradient accumulation steps')
    parser.add_argument('--patience', type=int, default=cfg.patience,
                        help='Early stopping patience')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision')
    parser.add_argument('--no_ema', action='store_true',
                        help='Disable Exponential Moving Average')
    parser.add_argument('--validate_every', type=int, default=cfg.validate_every,
                        help='Validate every N epochs')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=cfg.num_workers,
                        help='DataLoader workers')
    parser.add_argument('--save_dir', type=str, default=cfg.save_dir,
                        help='Save directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get training HDF5 files
    train_dir = Path(args.train_dir)
    train_files = sorted([str(f) for f in train_dir.glob('*.hdf5')])
    print(f"Found {len(train_files)} training HDF5 files")
    
    # Get validation HDF5 files
    val_dir = Path(args.val_dir)
    val_files = sorted([str(f) for f in val_dir.glob('*.hdf5')])
    print(f"Found {len(val_files)} validation HDF5 files")
    
    # Create datasets - now each sample is a single streamline
    train_dataset = StreamlineDataset(
        train_files,
        sampling_percentage=args.sampling_pct,
        max_streamlines_per_tract=args.max_streamlines_per_tract,
        full_sample_threshold=args.full_sample_threshold
    )
    val_dataset = StreamlineDataset(
        val_files,
        sampling_percentage=args.sampling_pct,
        max_streamlines_per_tract=args.max_streamlines_per_tract,
        full_sample_threshold=args.full_sample_threshold
    )
    
    print(f"\nTrain samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create stratified epoch sampler for training (proportional sampling each epoch)
    train_sampler = StratifiedEpochSampler(
        train_dataset,
        sampling_percentage=args.epoch_sampling_pct,
        min_samples_per_class=args.min_samples_per_class,
        full_sample_threshold=args.full_sample_threshold,
        seed=42,
        shuffle=True
    )
    
    # Create stratified validation sampler (20% per class for reliable Macro F1)
    val_sampler = StratifiedEpochSampler(
        val_dataset,
        sampling_percentage=0.10,  
        min_samples_per_class=10,
        full_sample_threshold=args.full_sample_threshold,
        seed=42,
        shuffle=False  # Keep validation deterministic
    )
    
    # Create dataloaders - batch_size now refers to number of streamlines
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Must be False when using sampler
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=streamline_collate_fn,
        persistent_workers=False,  # Disable to free RAM between epochs
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,  # Use subset sampler for faster validation
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=streamline_collate_fn,
        persistent_workers=False,  # Disable to free RAM between epochs
        prefetch_factor=2
    )
    
    # Create model
    if args.encoder_type == 'transformer':
        model = StreamlineEncoder(
            input_size=cfg.input_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            num_classes=args.num_classes,
            dropout=args.dropout,
            pooling=args.pooling
        )
    else:
        model = LightweightStreamlineEncoder(
            input_size=cfg.input_size,
            hidden_size=args.d_model,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            dropout=args.dropout
        )
    
    print(f"\nModel: {args.encoder_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Compile model for faster execution (PyTorch 2.0+)
    # Use dynamic=True for variable-length streamline inputs
    model = torch.compile(model, dynamic=True)
    print("Model compiled with torch.compile(dynamic=True)")
    
    # Calculate LR with optional batch size scaling
    if args.no_lr_scaling:
        lr = args.base_lr
        print(f"Using base LR: {lr:.2e}")
    else:
        # Linear scaling rule: LR scales with sqrt(batch_size / 256)
        lr = args.base_lr * math.sqrt(args.batch_size / 256)
        print(f"LR scaled with batch size: {args.base_lr:.2e} * sqrt({args.batch_size}/256) = {lr:.2e}")
    
    # Architecture-specific weight decay
    if args.encoder_type == 'transformer':
        weight_decay = cfg.weight_decay_transformer
        print(f"Using Transformer weight decay: {weight_decay:.2e}")
    else:
        weight_decay = cfg.weight_decay_lstm
        print(f"Using LSTM weight decay: {weight_decay:.2e}")
    
    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=lr,
        weight_decay=weight_decay,
        save_dir=args.save_dir,
        patience=args.patience,
        use_amp=not args.no_amp,
        accumulation_steps=args.accumulation_steps,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        warmup_steps=cfg.warmup_steps,
        plateau_patience=cfg.plateau_patience,
        plateau_factor=cfg.plateau_factor,
        train_dataset=train_dataset,
        num_classes=args.num_classes,
        resume_checkpoint=args.resume,
        use_ema=not args.no_ema,
        ema_decay=cfg.ema_decay,
        validate_every=args.validate_every
    )
    
    # Log experiment results to CSV
    log_experiment(
        experiment_name=f"{args.encoder_type}_d{args.d_model}_L{args.num_layers}",
        encoder_type=args.encoder_type,
        history=history,
        params={
            'd_model': args.d_model,
            'num_layers': args.num_layers,
            'nhead': args.nhead,
            'lr': lr,
            'base_lr': args.base_lr,
            'lr_scaled': not args.no_lr_scaling,
            'batch_size': args.batch_size,
            'dropout': args.dropout,
            'pooling': args.pooling,
            'parameters': sum(p.numel() for p in model.parameters()),
            'dim_feedforward': args.dim_feedforward,
            'num_workers': args.num_workers,
            'warmup_steps': cfg.warmup_steps,
            'plateau_patience': cfg.plateau_patience,
            'plateau_factor': cfg.plateau_factor,
            'weight_decay': weight_decay,
            'use_ema': not args.no_ema,
        },
        csv_path="experiments.csv"
    )


if __name__ == '__main__':
    main()


"""
EXAMPLES OF USE:
# Train with default config values
python src/train.py

# Train with LSTM encoder
python src/train.py --encoder_type lstm

# Train with larger batch size and more epochs
python src/train.py --batch_size 512 --epochs 100

# Train with custom data directory
python src/train.py --data_dir /path/to/your/hdf5/files

# Disable mixed precision (if you have GPU issues)
python src/train.py --no_amp

# Use mean pooling instead of CLS token
python src/train.py --pooling mean
"""