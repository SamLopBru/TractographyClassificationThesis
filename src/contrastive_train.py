"""
Contrastive Pre-training for Streamline Encoders.

Stage 1 of two-stage training:
  1. Pre-train encoder with Supervised Contrastive Loss (SupCon)
  2. Fine-tune classifier on frozen/unfrozen encoder (use train.py --pretrained_encoder)

Reference: "Supervised Contrastive Learning" (Khosla et al., 2020)
           https://arxiv.org/abs/2004.11362

Usage:
    # Pre-train with defaults
    uv run src/contrastive_train.py

    # Pre-train with custom settings
    uv run src/contrastive_train.py --epochs 30 --batch_size 1024 --temperature 0.07

    # Then fine-tune classifier
    uv run src/train.py --pretrained_encoder checkpoints/contrastive/pretrained_encoder.pt
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import os
import sys
import time
import math
import argparse
import copy
import gc
from datetime import datetime
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.encoder import StreamlineEncoder, LightweightStreamlineEncoder, ProjectionHead
from src.dataloader import StreamlineDataset, StratifiedEpochSampler, streamline_collate_fn
from src.config import TrainConfig, DEFAULT_CONFIG


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon).
    
    Pulls together embeddings of samples from the same class and pushes apart
    embeddings of samples from different classes in a temperature-scaled 
    cosine similarity space.
    
    Reference: "Supervised Contrastive Learning" (Khosla et al., 2020)
    
    Args:
        temperature: Temperature scaling factor (lower = sharper distribution)
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute SupCon loss.
        
        Args:
            features: L2-normalized embeddings of shape (batch_size, projection_dim)
            labels: Class labels of shape (batch_size,)
        
        Returns:
            Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Compute pairwise cosine similarity (features are already L2-normalized)
        # sim_matrix[i, j] = cos_sim(features[i], features[j]) / temperature
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same class, excluding self)
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()  # (batch, batch)
        
        # Remove self-similarity from the diagonal
        self_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - self_mask
        
        # Number of positives per sample
        num_positives = positive_mask.sum(dim=1)  # (batch,)
        
        # For numerical stability, subtract max from sim_matrix
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()
        
        # Compute log-sum-exp of all pairs (excluding self)
        # exp_sim[i, j] = exp(sim(i, j) / temperature) for j != i
        exp_sim = torch.exp(sim_matrix) * (1 - self_mask)
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        
        # Compute log-probability of positive pairs
        # log_prob[i, j] = sim(i, j) / temperature - log(sum_k exp(sim(i, k) / temperature))
        log_prob = sim_matrix - log_sum_exp
        
        # Average log-probability over positive pairs
        # Only consider samples that have at least one positive pair
        has_positives = num_positives > 0
        
        if has_positives.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Mean of log-probabilities of positive pairs for each anchor
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / num_positives.clamp(min=1)
        
        # Loss is negative mean log-probability (averaged over valid anchors)
        loss = -mean_log_prob[has_positives].mean()
        
        return loss


def compute_contrastive_metrics(features: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute alignment and uniformity metrics for monitoring contrastive learning.
    
    - Alignment: Mean distance between positive pairs (lower = better)
    - Uniformity: How uniformly features are distributed on the hypersphere (lower = better)
    
    Reference: "Understanding Contrastive Representation Learning through 
               Alignment and Uniformity on the Hypersphere" (Wang & Isola, 2020)
    """
    with torch.no_grad():
        # Alignment: average pairwise distance between positive pairs
        labels_col = labels.unsqueeze(1)
        positive_mask = (labels_col == labels_col.T).float()
        self_mask = torch.eye(features.shape[0], device=features.device)
        positive_mask = positive_mask - self_mask
        
        # Pairwise squared distances
        dist_matrix = torch.cdist(features, features, p=2).pow(2)
        
        num_positives = positive_mask.sum()
        if num_positives > 0:
            alignment = (dist_matrix * positive_mask).sum() / num_positives
        else:
            alignment = torch.tensor(0.0)
        
        # Uniformity: log of average pairwise Gaussian potential
        # Only compute on a subsample if batch is large (for efficiency)
        n = min(features.shape[0], 512)
        features_sub = features[:n]
        sq_dist = torch.cdist(features_sub, features_sub, p=2).pow(2)
        mask = 1 - torch.eye(n, device=features.device)
        uniformity = torch.log(
            (torch.exp(-2 * sq_dist) * mask).sum() / (n * (n - 1) + 1e-12)
        )
    
    return {
        'alignment': alignment.item(),
        'uniformity': uniformity.item()
    }


def contrastive_train_epoch(
    encoder: nn.Module,
    projection_head: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: GradScaler = None,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    log_interval: int = 50,
    scheduler: optim.lr_scheduler._LRScheduler = None,
    ema_encoder: nn.Module = None,
    ema_decay: float = 0.999
) -> Dict[str, float]:
    """Train encoder + projection head for one epoch with SupCon loss."""
    encoder.train()
    projection_head.train()
    
    total_loss = torch.tensor(0.0, device=device)
    total_samples = 0
    all_features = []
    all_labels = []
    
    start_time = time.time()
    optimizer.zero_grad()
    
    for batch_idx, (streamlines, lengths, labels) in enumerate(dataloader):
        streamlines = streamlines.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if use_amp and scaler is not None:
            with autocast(device_type='cuda', dtype=torch.float16):
                embeddings = encoder.get_embeddings(streamlines, lengths=lengths)
                projections = projection_head(embeddings)
                loss = criterion(projections, labels) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(projection_head.parameters()),
                    max_norm=1.0
                )
                
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                else:
                    print(f"  ⚠ Skipping batch {batch_idx+1} (inf/nan gradients)")
                
                scaler.update()
                optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()
                
                # EMA update for encoder only
                if ema_encoder is not None:
                    with torch.no_grad():
                        for ema_p, model_p in zip(ema_encoder.parameters(), encoder.parameters()):
                            ema_p.data.mul_(ema_decay).add_(model_p.data, alpha=1 - ema_decay)
        else:
            embeddings = encoder.get_embeddings(streamlines, lengths=lengths)
            projections = projection_head(embeddings)
            loss = criterion(projections, labels) / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(projection_head.parameters()),
                    max_norm=1.0
                )
                
                if torch.isfinite(grad_norm):
                    optimizer.step()
                else:
                    print(f"  ⚠ Skipping batch {batch_idx+1} (inf/nan gradients)")
                
                optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()
                
                if ema_encoder is not None:
                    with torch.no_grad():
                        for ema_p, model_p in zip(ema_encoder.parameters(), encoder.parameters()):
                            ema_p.data.mul_(ema_decay).add_(model_p.data, alpha=1 - ema_decay)
        
        # Metrics accumulation
        with torch.no_grad():
            total_loss += loss.detach() * accumulation_steps * labels.size(0)
            total_samples += labels.size(0)
            
            # Collect features for alignment/uniformity (last batch only to save memory)
            if batch_idx == len(dataloader) - 1:
                all_features.append(projections.detach())
                all_labels.append(labels.detach())
        
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss.item() / total_samples
            elapsed = time.time() - start_time
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
    
    # Compute alignment/uniformity on last batch
    metrics = {'loss': total_loss.item() / total_samples}
    
    if all_features:
        features_cat = torch.cat(all_features)
        labels_cat = torch.cat(all_labels)
        au_metrics = compute_contrastive_metrics(features_cat, labels_cat)
        metrics.update(au_metrics)
    
    return metrics


@torch.no_grad()
def contrastive_validate(
    encoder: nn.Module,
    projection_head: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True
) -> Dict[str, float]:
    """Validate encoder with contrastive loss and alignment/uniformity metrics."""
    encoder.eval()
    projection_head.eval()
    
    total_loss = 0.0
    total_samples = 0
    all_features = []
    all_labels = []
    
    for streamlines, lengths, labels in dataloader:
        streamlines = streamlines.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                embeddings = encoder.get_embeddings(streamlines, lengths=lengths)
                projections = projection_head(embeddings)
                loss = criterion(projections, labels)
        else:
            embeddings = encoder.get_embeddings(streamlines, lengths=lengths)
            projections = projection_head(embeddings)
            loss = criterion(projections, labels)
        
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        
        # Collect features for metrics (limit to avoid OOM)
        if len(all_features) * projections.shape[0] < 10000:
            all_features.append(projections)
            all_labels.append(labels)
    
    metrics = {'loss': total_loss / total_samples}
    
    if all_features:
        features_cat = torch.cat(all_features)
        labels_cat = torch.cat(all_labels)
        au_metrics = compute_contrastive_metrics(features_cat, labels_cat)
        metrics.update(au_metrics)
    
    return metrics


def contrastive_train(
    encoder: nn.Module,
    projection_head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    temperature: float = 0.07,
    save_dir: str = "checkpoints/contrastive",
    patience: int = 10,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    train_sampler: StratifiedEpochSampler = None,
    val_sampler: StratifiedEpochSampler = None,
    warmup_steps: int = 500,
    use_ema: bool = True,
    ema_decay: float = 0.999,
    validate_every: int = 2,
    log_interval: int = 50
) -> Dict[str, List[float]]:
    """
    Full contrastive pre-training loop.
    
    Trains the encoder + projection head with SupConLoss. 
    Saves the encoder weights (without projection head) for downstream fine-tuning.
    """
    encoder = encoder.to(device)
    projection_head = projection_head.to(device)
    
    # Mixed precision
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision (FP16) training")
    
    # Loss
    criterion = SupConLoss(temperature=temperature)
    print(f"Using SupConLoss with temperature={temperature}")
    
    # Optimizer (both encoder and projection head)
    params = list(encoder.parameters()) + list(projection_head.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    # Scheduler: warmup + cosine annealing
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    
    def warmup_cosine_fn(current_step):
        if current_step < warmup_steps:
            return 0.1 + 0.9 * (current_step / warmup_steps)
        else:
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_fn)
    
    print(f"Scheduler: {warmup_steps} steps warmup + cosine annealing (total {total_steps} steps)")
    
    # EMA encoder (no projection head in EMA — we only save the encoder)
    ema_encoder = None
    if use_ema:
        ema_encoder = copy.deepcopy(encoder)
        ema_encoder.eval()
        for p in ema_encoder.parameters():
            p.requires_grad = False
        print(f"Using EMA with decay={ema_decay}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # TensorBoard
    run_name = f"contrastive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join('runs', run_name))
    print(f"TensorBoard logs: runs/{run_name}")
    
    history = {
        'train_loss': [], 'val_loss': [],
        'alignment': [], 'uniformity': [],
        'epoch_time': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print('='*60)
        
        epoch_start = time.time()
        
        # Train
        train_metrics = contrastive_train_epoch(
            encoder, projection_head, train_loader, criterion,
            optimizer, device, epoch,
            scaler=scaler, use_amp=use_amp,
            accumulation_steps=accumulation_steps,
            log_interval=log_interval,
            scheduler=scheduler,
            ema_encoder=ema_encoder,
            ema_decay=ema_decay
        )
        
        print(f"\n  Train Loss: {train_metrics['loss']:.4f}")
        if 'alignment' in train_metrics:
            print(f"  Alignment: {train_metrics['alignment']:.4f} | Uniformity: {train_metrics['uniformity']:.4f}")
        
        # Validation
        should_validate = (epoch % validate_every == 0) or (epoch == epochs)
        
        if should_validate:
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            
            val_metrics = contrastive_validate(
                encoder, projection_head, val_loader, criterion, device, use_amp
            )
            
            # Also validate EMA encoder
            if ema_encoder is not None:
                ema_val = contrastive_validate(
                    ema_encoder, projection_head, val_loader, criterion, device, use_amp
                )
                print(f"  [EMA] Val Loss: {ema_val['loss']:.4f}")
        else:
            val_metrics = {'loss': history['val_loss'][-1] if history['val_loss'] else 0,
                          'alignment': 0, 'uniformity': 0}
            print("  [Skipping validation this epoch]")
        
        epoch_time = time.time() - epoch_start
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Time: {epoch_time:.1f}s")
        
        # History
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['alignment'].append(val_metrics.get('alignment', 0))
        history['uniformity'].append(val_metrics.get('uniformity', 0))
        history['epoch_time'].append(epoch_time)
        
        # TensorBoard
        writer.add_scalars('Contrastive_Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        if 'alignment' in val_metrics:
            writer.add_scalar('Alignment', val_metrics['alignment'], epoch)
            writer.add_scalar('Uniformity', val_metrics['uniformity'], epoch)
        
        # Save best encoder (based on validation loss)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            # Save encoder weights only (no projection head, no classifier)
            encoder_state = encoder.state_dict()
            # Remove _orig_mod prefix from torch.compile if present
            encoder_state = {k.replace('_orig_mod.', ''): v for k, v in encoder_state.items()}
            
            torch.save({
                'encoder_state_dict': encoder_state,
                'epoch': epoch,
                'val_loss': best_val_loss,
                'temperature': temperature,
                'history': history
            }, os.path.join(save_dir, 'pretrained_encoder.pt'))
            
            # Also save EMA encoder if available
            if ema_encoder is not None:
                ema_state = ema_encoder.state_dict()
                ema_state = {k.replace('_orig_mod.', ''): v for k, v in ema_state.items()}
                torch.save({
                    'encoder_state_dict': ema_state,
                    'epoch': epoch,
                    'val_loss': best_val_loss,
                }, os.path.join(save_dir, 'pretrained_encoder_ema.pt'))
            
            print(f"  ✓ Saved best pre-trained encoder (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping after {epoch} epochs")
                break
        
        # Memory cleanup
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"Contrastive pre-training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Pre-trained encoder saved to: {os.path.join(save_dir, 'pretrained_encoder.pt')}")
    print(f"Use with: uv run src/train.py --pretrained_encoder {os.path.join(save_dir, 'pretrained_encoder.pt')}")
    print('='*60)
    
    return history


def main():
    cfg = DEFAULT_CONFIG
    
    parser = argparse.ArgumentParser(description='Contrastive Pre-training for Streamline Encoder')
    
    # Data arguments
    parser.add_argument('--train_dir', type=str, default=cfg.train_dir)
    parser.add_argument('--val_dir', type=str, default=cfg.val_dir)
    parser.add_argument('--sampling_pct', type=float, default=cfg.sampling_pct)
    parser.add_argument('--epoch_sampling_pct', type=float, default=cfg.epoch_sampling_pct)
    parser.add_argument('--val_sampling_pct', type=float, default=cfg.val_sampling_pct)
    parser.add_argument('--min_samples_per_class', type=int, default=cfg.min_samples_per_class)
    parser.add_argument('--full_sample_threshold', type=int, default=cfg.full_sample_threshold)
    parser.add_argument('--max_streamlines_per_tract', type=int, default=cfg.max_streamlines_per_tract)
    
    # Encoder arguments
    parser.add_argument('--encoder_type', type=str, default=cfg.encoder_type,
                        choices=['transformer', 'lstm'])
    parser.add_argument('--d_model', type=int, default=cfg.d_model)
    parser.add_argument('--nhead', type=int, default=cfg.nhead)
    parser.add_argument('--num_layers', type=int, default=cfg.num_layers)
    parser.add_argument('--dim_feedforward', type=int, default=cfg.dim_feedforward)
    parser.add_argument('--dropout', type=float, default=cfg.dropout)
    parser.add_argument('--pooling', type=str, default='cls',
                        choices=['cls', 'mean', 'max', 'last'])
    
    # Contrastive-specific arguments
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='SupCon temperature (lower = sharper, default 0.07)')
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='Projection head output dimension')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size)
    parser.add_argument('--base_lr', type=float, default=cfg.base_lr)
    parser.add_argument('--no_lr_scaling', action='store_true')
    parser.add_argument('--accumulation_steps', type=int, default=cfg.accumulation_steps)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--no_ema', action='store_true')
    parser.add_argument('--validate_every', type=int, default=cfg.validate_every)
    parser.add_argument('--warmup_steps', type=int, default=cfg.warmup_steps)
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=cfg.num_workers)
    parser.add_argument('--save_dir', type=str, default='checkpoints/contrastive')
    parser.add_argument('--log_interval', type=int, default=cfg.log_interval)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enable TF32 for faster matmul on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Load data
    train_dir = Path(args.train_dir)
    train_files = sorted([str(f) for f in train_dir.glob('*.hdf5')])
    print(f"Found {len(train_files)} training HDF5 files")
    
    val_dir = Path(args.val_dir)
    val_files = sorted([str(f) for f in val_dir.glob('*.hdf5')])
    print(f"Found {len(val_files)} validation HDF5 files")
    
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
    
    # Samplers
    train_sampler = StratifiedEpochSampler(
        train_dataset,
        sampling_percentage=args.epoch_sampling_pct,
        min_samples_per_class=args.min_samples_per_class,
        full_sample_threshold=args.full_sample_threshold,
        seed=42, shuffle=True
    )
    
    val_sampler = StratifiedEpochSampler(
        val_dataset,
        sampling_percentage=args.val_sampling_pct,
        min_samples_per_class=10,
        full_sample_threshold=args.full_sample_threshold,
        seed=42, shuffle=False
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=True, collate_fn=streamline_collate_fn,
        persistent_workers=False, prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, num_workers=args.num_workers,
        pin_memory=True, collate_fn=streamline_collate_fn,
        persistent_workers=False, prefetch_factor=2
    )
    
    # Create encoder (without classifier — we only use get_embeddings())
    if args.encoder_type == 'transformer':
        encoder = StreamlineEncoder(
            input_size=cfg.input_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            num_classes=cfg.num_classes,  # Required by __init__, but classifier won't be used
            dropout=args.dropout,
            pooling=args.pooling
        )
        embedding_dim = args.d_model
    else:
        encoder = LightweightStreamlineEncoder(
            input_size=cfg.input_size,
            hidden_size=args.d_model,
            num_layers=args.num_layers,
            num_classes=cfg.num_classes,
            dropout=args.dropout,
            pooling=args.pooling if args.pooling in ['last', 'cls', 'mean', 'max'] else 'last'
        )
        embedding_dim = args.d_model * 2  # Bidirectional
    
    # Create projection head
    projection_head = ProjectionHead(
        input_dim=embedding_dim,
        hidden_dim=args.d_model,
        output_dim=args.projection_dim
    )
    
    total_params = sum(p.numel() for p in encoder.parameters()) + \
                   sum(p.numel() for p in projection_head.parameters())
    encoder_params = sum(p.numel() for p in encoder.parameters())
    
    print(f"\nEncoder: {args.encoder_type} ({encoder_params:,} params)")
    print(f"Projection Head: {sum(p.numel() for p in projection_head.parameters()):,} params")
    print(f"Total: {total_params:,} params")
    
    # Compile encoder for speed
    encoder = torch.compile(encoder, dynamic=True)
    print("Encoder compiled with torch.compile(dynamic=True)")
    
    # LR scaling
    if args.no_lr_scaling:
        lr = args.base_lr
    else:
        effective_batch_size = args.batch_size * args.accumulation_steps
        lr = args.base_lr * math.sqrt(effective_batch_size / 256)
        print(f"LR scaled: {args.base_lr:.2e} * sqrt({effective_batch_size}/256) = {lr:.2e}")
    
    # Weight decay
    weight_decay = cfg.weight_decay_transformer if args.encoder_type == 'transformer' else cfg.weight_decay_lstm
    
    # Train
    history = contrastive_train(
        encoder=encoder,
        projection_head=projection_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=lr,
        weight_decay=weight_decay,
        temperature=args.temperature,
        save_dir=args.save_dir,
        patience=args.patience,
        use_amp=not args.no_amp,
        accumulation_steps=args.accumulation_steps,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        warmup_steps=args.warmup_steps,
        use_ema=not args.no_ema,
        ema_decay=cfg.ema_decay,
        validate_every=args.validate_every,
        log_interval=args.log_interval
    )


if __name__ == '__main__':
    main()
