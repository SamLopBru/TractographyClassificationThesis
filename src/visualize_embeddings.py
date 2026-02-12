"""
t-SNE / UMAP Visualization of Streamline Encoder Embeddings.

Extracts pooled representations from a trained encoder checkpoint and
generates 2D scatter plots colored by class label.

Works with both classification checkpoints (best_model.pt) and
contrastive pre-training checkpoints (pretrained_encoder.pt).

Usage:
    # Visualize from a classification checkpoint
    uv run src/visualize_embeddings.py --checkpoint checkpoints/best_model.pt --data_dir sequences/validset

    # Visualize from a contrastive pre-training checkpoint
    uv run src/visualize_embeddings.py --checkpoint checkpoints/contrastive/pretrained_encoder.pt --data_dir sequences/validset

    # Limit samples for faster visualization
    uv run src/visualize_embeddings.py --checkpoint checkpoints/best_model.pt --max_samples 5000

    # Use UMAP instead of t-SNE (requires umap-learn package)
    uv run src/visualize_embeddings.py --checkpoint checkpoints/best_model.pt --method umap
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from pathlib import Path
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.encoder import StreamlineEncoder, LightweightStreamlineEncoder
from src.dataloader import StreamlineDataset, streamline_collate_fn
from src.config import TrainConfig, DEFAULT_CONFIG


def load_encoder(
    checkpoint_path: str,
    config: TrainConfig,
    device: torch.device,
    encoder_type: str = 'transformer',
    pooling: str = 'cls',
    d_model: int = None,
    num_layers: int = None
) -> nn.Module:
    """
    Load encoder from either a classification or contrastive checkpoint.
    
    Automatically detects checkpoint type by checking for 'encoder_state_dict'
    (contrastive) vs 'model_state_dict' (classification).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Use provided or config defaults
    _d_model = d_model or config.d_model
    _num_layers = num_layers or config.num_layers
    
    # Create encoder
    if encoder_type == 'transformer':
        encoder = StreamlineEncoder(
            input_size=config.input_size,
            d_model=_d_model,
            nhead=config.nhead,
            num_layers=_num_layers,
            dim_feedforward=config.dim_feedforward,
            num_classes=config.num_classes,
            dropout=config.dropout,
            pooling=pooling
        )
    else:
        encoder = LightweightStreamlineEncoder(
            input_size=config.input_size,
            hidden_size=_d_model,
            num_layers=_num_layers,
            num_classes=config.num_classes,
            dropout=config.dropout,
            pooling=pooling if pooling in ['last', 'cls', 'mean', 'max'] else 'last'
        )
    
    # Detect checkpoint type and load weights
    if 'encoder_state_dict' in checkpoint:
        # Contrastive pre-training checkpoint
        state_dict = checkpoint['encoder_state_dict']
        print(f"Loaded contrastive checkpoint (epoch {checkpoint.get('epoch', '?')})")
    elif 'model_state_dict' in checkpoint:
        # Classification checkpoint
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded classification checkpoint (epoch {checkpoint.get('epoch', '?')})")
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")
    
    # Handle torch.compile prefix
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    # Load with strict=False to handle classifier weights in classification checkpoints
    encoder.load_state_dict(state_dict, strict=False)
    encoder = encoder.to(device)
    encoder.eval()
    
    return encoder


@torch.no_grad()
def extract_embeddings(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None,
    use_amp: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract pooled embeddings from the encoder.
    
    Args:
        encoder: Trained encoder model
        dataloader: DataLoader providing (streamlines, lengths, labels)
        device: Device to run inference on
        max_samples: Maximum number of samples to extract (None = all)
        use_amp: Whether to use mixed precision
    
    Returns:
        embeddings: Array of shape (n_samples, embedding_dim)
        labels: Array of shape (n_samples,)
    """
    encoder.eval()
    
    all_embeddings = []
    all_labels = []
    total = 0
    
    print("Extracting embeddings...")
    for batch_idx, (streamlines, lengths, labels) in enumerate(dataloader):
        streamlines = streamlines.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        
        if use_amp:
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=torch.float16):
                embeddings = encoder.get_embeddings(streamlines, lengths=lengths)
        else:
            embeddings = encoder.get_embeddings(streamlines, lengths=lengths)
        
        all_embeddings.append(embeddings.float().cpu().numpy())
        all_labels.append(labels.numpy())
        
        total += labels.size(0)
        if max_samples is not None and total >= max_samples:
            break
        
        if (batch_idx + 1) % 20 == 0:
            print(f"  Processed {total} samples...")
    
    embeddings = np.concatenate(all_embeddings)
    labels = np.concatenate(all_labels)
    
    # Trim to max_samples
    if max_samples is not None and len(embeddings) > max_samples:
        embeddings = embeddings[:max_samples]
        labels = labels[:max_samples]
    
    print(f"Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    return embeddings, labels


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    perplexity: float = 30.0,
    title: str = "t-SNE of Encoder Embeddings",
    class_names: Optional[list] = None,
    figsize: tuple = (14, 12)
):
    """
    Generate and save a 2D t-SNE scatter plot.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        labels: Array of shape (n_samples,)
        save_path: Path to save the plot
        perplexity: t-SNE perplexity parameter
        title: Plot title
        class_names: Optional list of class names for legend
        figsize: Figure size
    """
    print(f"Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000,
        learning_rate='auto',
        init='pca'
    )
    embeddings_2d = tsne.fit_transform(embeddings)
    print("t-SNE complete.")
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Use a colormap with enough distinct colors
    if n_classes <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_ncar
    
    colors = [cmap(i / n_classes) for i in range(n_classes)]
    
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if class_names else f"Class {label}"
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[idx]],
            label=name,
            s=3,
            alpha=0.6,
            rasterized=True  # For faster rendering with many points
        )
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    # Legend outside the plot
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=8,
        markerscale=3,
        framealpha=0.9
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved t-SNE plot to {save_path}")


def plot_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    title: str = "UMAP of Encoder Embeddings",
    class_names: Optional[list] = None,
    figsize: tuple = (14, 12)
):
    """
    Generate and save a 2D UMAP scatter plot.
    
    Requires the umap-learn package. Falls back to t-SNE if not installed.
    """
    try:
        import umap
    except ImportError:
        print("umap-learn not installed. Falling back to t-SNE.")
        print("Install with: pip install umap-learn")
        plot_tsne(embeddings, labels, save_path, title=title.replace("UMAP", "t-SNE"),
                  class_names=class_names, figsize=figsize)
        return
    
    print(f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        metric='cosine'
    )
    embeddings_2d = reducer.fit_transform(embeddings)
    print("UMAP complete.")
    
    # Same plotting as t-SNE
    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    cmap = plt.cm.tab20 if n_classes <= 20 else plt.cm.gist_ncar
    colors = [cmap(i / n_classes) for i in range(n_classes)]
    
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if class_names else f"Class {label}"
        ax.scatter(
            embeddings_2d[mask, 0], embeddings_2d[mask, 1],
            c=[colors[idx]], label=name, s=3, alpha=0.6, rasterized=True
        )
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8,
              markerscale=3, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved UMAP plot to {save_path}")


def main():
    cfg = DEFAULT_CONFIG
    
    parser = argparse.ArgumentParser(description='Visualize Encoder Embeddings')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint (classification or contrastive)')
    parser.add_argument('--data_dir', type=str, default='sequences/validset',
                        help='Directory containing HDF5 files to visualize')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Directory to save plots')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='Maximum number of samples to visualize')
    parser.add_argument('--method', type=str, default='tsne',
                        choices=['tsne', 'umap', 'both'],
                        help='Dimensionality reduction method')
    parser.add_argument('--perplexity', type=float, default=30.0,
                        help='t-SNE perplexity parameter')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size for embedding extraction')
    parser.add_argument('--num_workers', type=int, default=cfg.num_workers)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--sampling_pct', type=float, default=0.25,
                        help='Percentage of data to index')
    
    # Encoder config (must match checkpoint)
    parser.add_argument('--encoder_type', type=str, default=cfg.encoder_type,
                        choices=['transformer', 'lstm'])
    parser.add_argument('--pooling', type=str, default='cls',
                        choices=['cls', 'mean', 'max', 'last'])
    parser.add_argument('--d_model', type=int, default=None,
                        help='Model dimension (default: from config)')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Number of layers (default: from config)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load encoder
    encoder = load_encoder(
        args.checkpoint, cfg, device,
        encoder_type=args.encoder_type,
        pooling=args.pooling,
        d_model=args.d_model,
        num_layers=args.num_layers
    )
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Load data
    data_dir = Path(args.data_dir)
    data_files = sorted([str(f) for f in data_dir.glob('*.hdf5')])
    print(f"\nFound {len(data_files)} HDF5 files in {data_dir}")
    
    if not data_files:
        print("ERROR: No HDF5 files found!")
        return
    
    dataset = StreamlineDataset(
        data_files,
        sampling_percentage=args.sampling_pct,
        full_sample_threshold=100000
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=streamline_collate_fn
    )
    
    # Extract embeddings
    embeddings, labels = extract_embeddings(
        encoder, dataloader, device,
        max_samples=args.max_samples,
        use_amp=not args.no_amp
    )
    
    # Class names
    class_names = [f"Bundle_{i}" for i in range(cfg.num_classes)]
    
    # Generate checkpoint name for title
    ckpt_name = Path(args.checkpoint).stem
    
    # Generate plots
    if args.method in ['tsne', 'both']:
        plot_tsne(
            embeddings, labels,
            save_path=os.path.join(args.output_dir, f'tsne_{ckpt_name}.png'),
            perplexity=args.perplexity,
            title=f"t-SNE — {ckpt_name} ({len(embeddings):,} samples)",
            class_names=class_names
        )
    
    if args.method in ['umap', 'both']:
        plot_umap(
            embeddings, labels,
            save_path=os.path.join(args.output_dir, f'umap_{ckpt_name}.png'),
            title=f"UMAP — {ckpt_name} ({len(embeddings):,} samples)",
            class_names=class_names
        )
    
    # Save raw embeddings for further analysis
    np.savez_compressed(
        os.path.join(args.output_dir, f'embeddings_{ckpt_name}.npz'),
        embeddings=embeddings,
        labels=labels
    )
    print(f"\nSaved raw embeddings to {args.output_dir}/embeddings_{ckpt_name}.npz")
    
    print(f"\n{'='*60}")
    print(f"Visualization complete! Results in: {args.output_dir}/")
    print('='*60)


if __name__ == '__main__':
    main()
