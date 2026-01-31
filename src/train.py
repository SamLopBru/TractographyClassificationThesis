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
from typing import Dict, List, Tuple
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.encoder import StreamlineEncoder, LightweightStreamlineEncoder
from src.dataloader import StreamlineDataset, streamline_collate_fn
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
    log_interval: int = 10
) -> Dict[str, float]:
    """Train for one epoch with optional mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    start_time = time.time()
    optimizer.zero_grad()
    
    for batch_idx, (streamlines, lengths, labels) in enumerate(dataloader):
        streamlines = streamlines.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        
        # Forward pass with optional mixed precision
        if use_amp and scaler is not None:
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(streamlines, lengths=lengths)
                loss = criterion(logits, labels) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(streamlines, lengths=lengths)
            loss = criterion(logits, labels) / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
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
        'accuracy': 100.0 * total_correct / total_samples
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for streamlines, lengths, labels in dataloader:
        streamlines = streamlines.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        
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
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': 100.0 * total_correct / total_samples
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
    accumulation_steps: int = 1
) -> Dict[str, List[float]]:
    """
    Full training loop with validation, early stopping, and mixed precision.
    """
    model = model.to(device)
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision (FP16) training")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs} | LR: {scheduler.get_last_lr()[0]:.2e}")
        print('='*60)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler, use_amp=use_amp,
            accumulation_steps=accumulation_steps
        )
        print(f"\n  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, use_amp=use_amp)
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'history': history
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
            print(f"  ✓ Saved new best model (Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping triggered after {epoch} epochs")
                break
        
        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }, os.path.join(save_dir, 'latest_checkpoint.pt'))
    
    # Save training history
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print('='*60)
    
    return history


def main():
    # Use DEFAULT_CONFIG values as defaults for argparse
    cfg = DEFAULT_CONFIG
    
    parser = argparse.ArgumentParser(description='Train Streamline Bundle Classifier')
    
    # Data arguments
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Directory containing training HDF5 files')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Directory containing validation HDF5 files')
    parser.add_argument('--sampling_pct', type=float, default=cfg.sampling_pct,
                        help='Percentage of streamlines to sample per tract')
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
    parser.add_argument('--lr', type=float, default=cfg.lr,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=cfg.weight_decay,
                        help='Weight decay')
    parser.add_argument('--accumulation_steps', type=int, default=cfg.accumulation_steps,
                        help='Gradient accumulation steps')
    parser.add_argument('--patience', type=int, default=cfg.patience,
                        help='Early stopping patience')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=cfg.num_workers,
                        help='DataLoader workers')
    parser.add_argument('--save_dir', type=str, default=cfg.save_dir,
                        help='Save directory')
    
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
        max_streamlines_per_tract=args.max_streamlines_per_tract
    )
    val_dataset = StreamlineDataset(
        val_files,
        sampling_percentage=args.sampling_pct,
        max_streamlines_per_tract=args.max_streamlines_per_tract
    )
    
    print(f"\nTrain samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders - batch_size now refers to number of streamlines
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=streamline_collate_fn,
        persistent_workers=args.num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=streamline_collate_fn,
        persistent_workers=args.num_workers > 0
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
    
    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        patience=args.patience,
        use_amp=not args.no_amp,
        accumulation_steps=args.accumulation_steps
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