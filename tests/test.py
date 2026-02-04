"""
Test script for evaluating the best_model.pt checkpoint.

This script computes various classification metrics and generates
visualization plots to assess model performance.

Usage:
    python test.py --test_dir sequences/testset
    python test.py --test_dir sequences/validset --checkpoint checkpoints/latest_checkpoint.pt
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from pathlib import Path
import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Metrics from sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./")))

from src.encoder import StreamlineEncoder, LightweightStreamlineEncoder
from src.dataloader import StreamlineDataset, streamline_collate_fn
from src.config import TrainConfig, DEFAULT_CONFIG


def load_model(
    checkpoint_path: str,
    config: TrainConfig,
    device: torch.device
) -> nn.Module:
    """Load a trained model from checkpoint."""
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model based on config
    if config.encoder_type == 'transformer':
        model = StreamlineEncoder(
            input_size=config.input_size,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            num_classes=config.num_classes,
            dropout=config.dropout,
            pooling=config.pooling
        )
    else:
        model = LightweightStreamlineEncoder(
            input_size=config.input_size,
            hidden_size=config.d_model,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_accuracy' in checkpoint:
        print(f"Checkpoint validation accuracy: {checkpoint['val_accuracy']:.2f}%")
    
    return model, checkpoint


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on the test set.
    
    Returns:
        all_labels: Ground truth labels
        all_preds: Predicted labels
        all_probs: Softmax probabilities for all classes
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    print("Running inference...")
    for batch_idx, (streamlines, lengths, labels) in enumerate(dataloader):
        streamlines = streamlines.to(device)
        lengths = lengths.to(device)
        
        if use_amp:
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=torch.float16):
                logits = model(streamlines, lengths=lengths)
        else:
            logits = model(streamlines, lengths=lengths)
        
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_labels.append(labels.numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    
    print(f"Total samples evaluated: {len(all_labels)}")
    
    return all_labels, all_preds, all_probs


def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
    class_names: Optional[List[str]] = None
) -> Dict:
    """Compute comprehensive classification metrics."""
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(labels, preds) * 100
    metrics['precision_macro'] = precision_score(labels, preds, average='macro', zero_division=0) * 100
    metrics['precision_weighted'] = precision_score(labels, preds, average='weighted', zero_division=0) * 100
    metrics['recall_macro'] = recall_score(labels, preds, average='macro', zero_division=0) * 100
    metrics['recall_weighted'] = recall_score(labels, preds, average='weighted', zero_division=0) * 100
    metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0) * 100
    metrics['f1_weighted'] = f1_score(labels, preds, average='weighted', zero_division=0) * 100
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(labels, preds, average=None, zero_division=0) * 100
    metrics['recall_per_class'] = recall_score(labels, preds, average=None, zero_division=0) * 100
    metrics['f1_per_class'] = f1_score(labels, preds, average=None, zero_division=0) * 100
    
    # Top-k accuracy (if more than 2 classes)
    if num_classes > 2:
        for k in [3, 5]:
            if k <= num_classes:
                metrics[f'top_{k}_accuracy'] = top_k_accuracy_score(labels, probs, k=k) * 100
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(labels, preds)
    
    # Classification report
    target_names = class_names if class_names else [f"Class {i}" for i in range(num_classes)]
    present_classes = np.unique(np.concatenate([labels, preds]))
    target_names_present = [target_names[i] for i in present_classes]
    metrics['classification_report'] = classification_report(
        labels, preds, 
        target_names=target_names_present, 
        zero_division=0
    )
    
    # Sample counts per class
    unique, counts = np.unique(labels, return_counts=True)
    metrics['samples_per_class'] = dict(zip(unique.tolist(), counts.tolist()))
    
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (14, 12)
):
    """Plot and save confusion matrix."""
    
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    else:
        cm_normalized = cm
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use seaborn heatmap for better visualization
    sns.heatmap(
        cm_normalized,
        annot=True if cm.shape[0] <= 20 else False,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names if class_names else range(cm.shape[1]),
        yticklabels=class_names if class_names else range(cm.shape[0]),
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix (Normalized)' if normalize else 'Confusion Matrix', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_per_class_metrics(
    metrics: Dict,
    save_path: str,
    class_names: Optional[List[str]] = None
):
    """Plot per-class precision, recall, and F1 scores."""
    
    n_classes = len(metrics['precision_per_class'])
    x = np.arange(n_classes)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(max(12, n_classes * 0.5), 6))
    
    rects1 = ax.bar(x - width, metrics['precision_per_class'], width, label='Precision', color='#2ecc71')
    rects2 = ax.bar(x, metrics['recall_per_class'], width, label='Recall', color='#3498db')
    rects3 = ax.bar(x + width, metrics['f1_per_class'], width, label='F1 Score', color='#e74c3c')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names if class_names else [f'{i}' for i in range(n_classes)], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    # Add horizontal line for macro average
    ax.axhline(y=metrics['f1_macro'], color='#9b59b6', linestyle='--', label=f"Macro F1: {metrics['f1_macro']:.1f}%")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class metrics plot to {save_path}")


def plot_training_history(
    history: Dict,
    save_path: str
):
    """Plot training history curves."""
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch = np.argmax(history['val_acc']) + 1
    best_acc = max(history['val_acc'])
    axes[1].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
    axes[1].annotate(f'Best: {best_acc:.1f}%\nEpoch {best_epoch}', 
                     xy=(best_epoch, best_acc), 
                     xytext=(best_epoch + 1, best_acc - 5),
                     fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training history plot to {save_path}")


def plot_class_distribution(
    samples_per_class: Dict,
    save_path: str,
    class_names: Optional[List[str]] = None
):
    """Plot class distribution in the test set."""
    
    classes = sorted(samples_per_class.keys())
    counts = [samples_per_class[c] for c in classes]
    
    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.4), 6))
    
    bars = ax.bar(range(len(classes)), counts, color='#3498db', edgecolor='#2980b9')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Test Set Class Distribution', fontsize=14)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(class_names if class_names else [f'{c}' for c in classes], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{count}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved class distribution plot to {save_path}")


def plot_prediction_confidence(
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    save_path: str
):
    """Plot confidence distribution for correct vs incorrect predictions."""
    
    # Get max probability (confidence) for each prediction
    confidences = probs.max(axis=1)
    correct = (labels == preds)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of confidences
    axes[0].hist(confidences[correct], bins=50, alpha=0.7, label='Correct', color='#2ecc71', density=True)
    axes[0].hist(confidences[~correct], bins=50, alpha=0.7, label='Incorrect', color='#e74c3c', density=True)
    axes[0].set_xlabel('Prediction Confidence', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Confidence Distribution', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy by confidence bins
    confidence_bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(confidences, confidence_bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(confidence_bins) - 2)
    
    bin_accuracies = []
    bin_counts = []
    for i in range(len(confidence_bins) - 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracies.append(correct[mask].mean() * 100)
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
    
    ax2 = axes[1]
    ax2.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, color='#3498db', label='Accuracy')
    ax2.set_xlabel('Confidence Bin', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy by Confidence Level', fontsize=14)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    
    # Add sample counts as text
    for x, acc, count in zip(bin_centers, bin_accuracies, bin_counts):
        if count > 0:
            ax2.text(x, acc + 2, f'{count}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confidence analysis plot to {save_path}")


def plot_misclassification_analysis(
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: str,
    class_names: Optional[List[str]] = None,
    top_n: int = 10
):
    """Analyze and plot most common misclassifications as percentages."""
    
    total_samples = len(labels)
    
    # Find misclassifications
    mask = labels != preds
    misclassified_pairs = list(zip(labels[mask], preds[mask]))
    
    # Count each type of misclassification
    pair_counts = defaultdict(int)
    for true, pred in misclassified_pairs:
        pair_counts[(true, pred)] += 1
    
    # Calculate percentage of total samples for each misclassification type
    pair_percentages = {k: (v / total_samples) * 100 for k, v in pair_counts.items()}
    
    # Get top N most common misclassifications by percentage
    sorted_pairs = sorted(pair_percentages.items(), key=lambda x: -x[1])[:top_n]
    
    if not sorted_pairs:
        print("No misclassifications found!")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels_list = [f"{class_names[p[0]] if class_names else p[0]} → {class_names[p[1]] if class_names else p[1]}" 
                   for p, _ in sorted_pairs]
    percentages = [pct for _, pct in sorted_pairs]
    counts = [pair_counts[p] for p, _ in sorted_pairs]
    
    y_pos = np.arange(len(labels_list))
    bars = ax.barh(y_pos, percentages, color='#e74c3c', edgecolor='#c0392b')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_list)
    ax.invert_yaxis()
    ax.set_xlabel('Percentage of Total Samples (%)', fontsize=12)
    ax.set_title(f'Top {top_n} Most Common Misclassifications (% of Total)', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    # Add percentage and count labels
    for bar, pct, count in zip(bars, percentages, counts):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{pct:.2f}% ({count:,})', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved misclassification analysis plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test Streamline Bundle Classifier')
    
    # Use default config for defaults
    cfg = DEFAULT_CONFIG
    
    parser.add_argument('--test_dir', type=str, default='sequences/testset',
                        help='Directory containing test HDF5 files')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='tests/test_results',
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=cfg.num_workers,
                        help='DataLoader workers')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision')
    parser.add_argument('--sampling_pct', type=float, default=0.25,
                        help='Percentage of test data to use (1.0 = all)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model, checkpoint = load_model(args.checkpoint, cfg, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get test files
    test_dir = Path(args.test_dir)
    test_files = sorted([str(f) for f in test_dir.glob('*.hdf5')])
    print(f"\nFound {len(test_files)} test HDF5 files in {test_dir}")
    
    if not test_files:
        print("ERROR: No HDF5 files found in test directory!")
        return
    
    # Create test dataset (use all data by default for testing)
    test_dataset = StreamlineDataset(
        test_files,
        sampling_percentage=args.sampling_pct,
        full_sample_threshold=100000  # Effectively disable threshold for test
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=streamline_collate_fn
    )
    
    # Run evaluation
    labels, preds, probs = evaluate_model(model, test_loader, device, use_amp=not args.no_amp)
    
    # Get class names (if available, otherwise use class IDs)
    unique_classes = np.unique(np.concatenate([labels, preds]))
    class_names = [f"Bundle_{i}" for i in range(cfg.num_classes)]
    
    # Compute metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    metrics = compute_metrics(labels, preds, probs, cfg.num_classes, class_names)
    
    # Print summary
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:          {metrics['accuracy']:.2f}%")
    print(f"   Precision (macro): {metrics['precision_macro']:.2f}%")
    print(f"   Recall (macro):    {metrics['recall_macro']:.2f}%")
    print(f"   F1 Score (macro):  {metrics['f1_macro']:.2f}%")
    print(f"   F1 Score (weighted): {metrics['f1_weighted']:.2f}%")
    
    if 'top_3_accuracy' in metrics:
        print(f"   Top-3 Accuracy:    {metrics['top_3_accuracy']:.2f}%")
    if 'top_5_accuracy' in metrics:
        print(f"   Top-5 Accuracy:    {metrics['top_5_accuracy']:.2f}%")
    
    print(f"\n📋 Classification Report:\n")
    print(metrics['classification_report'])
    
    # Generate plots
    print("\n📈 Generating visualization plots...")
    
    # 1. Confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        os.path.join(args.output_dir, 'confusion_matrix.png'),
        class_names=class_names
    )
    
    # 2. Per-class metrics
    plot_per_class_metrics(
        metrics,
        os.path.join(args.output_dir, 'per_class_metrics.png'),
        class_names=class_names
    )
    
    # 3. Training history (if available)
    history_path = os.path.join(os.path.dirname(args.checkpoint), 'history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        plot_training_history(
            history,
            os.path.join(args.output_dir, 'training_history.png')
        )
    
    # 4. Class distribution
    plot_class_distribution(
        metrics['samples_per_class'],
        os.path.join(args.output_dir, 'class_distribution.png'),
        class_names=class_names
    )
    
    # 5. Confidence analysis
    plot_prediction_confidence(
        labels, probs, preds,
        os.path.join(args.output_dir, 'confidence_analysis.png')
    )
    
    # 6. Misclassification analysis
    plot_misclassification_analysis(
        labels, preds,
        os.path.join(args.output_dir, 'misclassification_analysis.png'),
        class_names=class_names
    )
    
    # Save metrics to JSON
    metrics_to_save = {
        'accuracy': metrics['accuracy'],
        'precision_macro': metrics['precision_macro'],
        'precision_weighted': metrics['precision_weighted'],
        'recall_macro': metrics['recall_macro'],
        'recall_weighted': metrics['recall_weighted'],
        'f1_macro': metrics['f1_macro'],
        'f1_weighted': metrics['f1_weighted'],
        'samples_per_class': metrics['samples_per_class'],
        'total_samples': len(labels),
        'total_correct': int((labels == preds).sum()),
        'total_incorrect': int((labels != preds).sum())
    }
    
    if 'top_3_accuracy' in metrics:
        metrics_to_save['top_3_accuracy'] = metrics['top_3_accuracy']
    if 'top_5_accuracy' in metrics:
        metrics_to_save['top_5_accuracy'] = metrics['top_5_accuracy']
    
    metrics_file = os.path.join(args.output_dir, 'test_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nSaved metrics to {metrics_file}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\n📁 All results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()


"""
EXAMPLES OF USE:

# Test on validation set with default settings
python test.py

# Test on a specific test directory
python test.py --test_dir sequences/testset

# Test with a specific checkpoint
python test.py --checkpoint checkpoints/latest_checkpoint.pt

# Test with smaller batch size (if memory issues)
python test.py --batch_size 256

# Disable mixed precision
python test.py --no_amp

# Test on a subset of data (10%)
python test.py --sampling_pct 0.1
"""
