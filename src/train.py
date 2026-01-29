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

from src.encoder import StreamlineEncoder, LightweightStreamlineEncoder, create_encoder
from src.dataloader import BalancedTractDataset, custom_collate


def prepare_batch(
    batch: List, 
    device: torch.device,
    max_streamlines: int = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare a batch from the dataloader for training.
    
    The dataloader returns a list of subjects, each containing tract dictionaries.
    We need to flatten this into a single batch of streamlines with labels.
    
    Args:
        batch: List of subject data from dataloader
        device: Target device
        max_streamlines: Maximum number of streamlines per batch (for memory management)
    
    Returns:
        streamlines: (total_streamlines, max_seq_len, 5)
        lengths: (total_streamlines,)
        labels: (total_streamlines,)
    """
    all_streamlines = []
    all_lengths = []
    all_labels = []
    
    # Flatten batch: iterate over subjects, then tracts
    for subject_data in batch:
        for tract_dict in subject_data:
            streamlines = tract_dict['streamlines']  # (n_streamlines, max_len, 5)
            lengths = tract_dict['lengths']          # (n_streamlines,)
            tract_id = tract_dict['tract_id']
            n_streamlines = tract_dict['n_streamlines']
            
            all_streamlines.append(streamlines)
            all_lengths.append(lengths)
            all_labels.append(torch.full((n_streamlines,), tract_id, dtype=torch.long))
    
    # Pad to same max length across all tracts in this batch
    max_len = max(s.shape[1] for s in all_streamlines)
    n_features = all_streamlines[0].shape[2]
    
    padded_streamlines = []
    for s in all_streamlines:
        if s.shape[1] < max_len:
            padding = torch.zeros(s.shape[0], max_len - s.shape[1], n_features)
            s = torch.cat([s, padding], dim=1)
        padded_streamlines.append(s)
    
    streamlines = torch.cat(padded_streamlines, dim=0)
    lengths = torch.cat(all_lengths, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Limit batch size to prevent OOM or CUDA attention errors (0 or None = auto-limit to 60000)
    # CUDA Flash Attention has a hard limit of 65535 batch size
    effective_max = max_streamlines if max_streamlines and max_streamlines > 0 else 60000
    if streamlines.size(0) > effective_max:
        # Randomly sample to keep batch size manageable
        indices = torch.randperm(streamlines.size(0))[:effective_max]
        streamlines = streamlines[indices]
        lengths = lengths[indices]
        labels = labels[indices]
    
    return streamlines.to(device), lengths.to(device), labels.to(device)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: GradScaler = None,
    use_amp: bool = True,
    max_streamlines: int = 2000,
    accumulation_steps: int = 1,
    log_interval: int = 10
) -> Dict[str, float]:
    """Train for one epoch with optional mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        streamlines, lengths, labels = prepare_batch(batch, device, max_streamlines)
        
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
    use_amp: bool = True,
    max_streamlines: int = 2000
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch in dataloader:
        streamlines, lengths, labels = prepare_batch(batch, device, max_streamlines)
        
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
    max_streamlines: int = 2000,
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
            scaler=scaler, use_amp=use_amp, max_streamlines=max_streamlines,
            accumulation_steps=accumulation_steps
        )
        print(f"\n  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, use_amp=use_amp, max_streamlines=max_streamlines)
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
    parser = argparse.ArgumentParser(description='Train Streamline Bundle Classifier')
    parser.add_argument('--data_dir', type=str, default='preprocessing/sequences',
                        help='Directory containing HDF5 files')
    parser.add_argument('--encoder_type', type=str, default='transformer',
                        choices=['transformer', 'lstm'], help='Type of encoder')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (subjects)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--sampling_pct', type=float, default=0.05,
                        help='Percentage of streamlines to sample per tract')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--max_streamlines', type=int, default=0,
                        help='Max streamlines per batch (0 = no limit)')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get all HDF5 files
    data_dir = Path(args.data_dir)
    all_files = sorted([str(f) for f in data_dir.glob('*.hdf5')])
    print(f"Found {len(all_files)} HDF5 files")
    
    # Train/val split
    n_val = max(1, int(len(all_files) * args.val_split))
    train_files = all_files[:-n_val]
    val_files = all_files[-n_val:]
    
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    
    # Create datasets
    train_dataset = BalancedTractDataset(
        train_files,
        sampling_percentage=args.sampling_pct,
        keep_padded=True
    )
    val_dataset = BalancedTractDataset(
        val_files,
        sampling_percentage=args.sampling_pct,
        keep_padded=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate,
        persistent_workers=args.num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate,
        persistent_workers=args.num_workers > 0
    )
    
    # Create model
    if args.encoder_type == 'transformer':
        model = StreamlineEncoder(
            input_size=5,
            d_model=args.d_model,
            nhead=8,
            num_layers=args.num_layers,
            num_classes=32,
            dropout=0.1
        )
    else:
        model = LightweightStreamlineEncoder(
            input_size=5,
            hidden_size=args.d_model,
            num_layers=args.num_layers,
            num_classes=32,
            dropout=0.1
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
        save_dir=args.save_dir,
        patience=args.patience,
        use_amp=not args.no_amp,
        max_streamlines=args.max_streamlines,
        accumulation_steps=args.accumulation_steps
    )


if __name__ == '__main__':
    main()
