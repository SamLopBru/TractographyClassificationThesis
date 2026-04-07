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
import pathlib
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.encoder import TransformerEncoder, LSTMEncoder
from utils.dataloader import StreamlineDataset, streamline_collate_fn
from src.config import TrainConfig, DEFAULT_CONFIG

# wDice evaluation imports (lazy-loaded in the wDice branch)
from wDice import (
    ENCODED_TRACTS, ID_TO_TRACT,
    run_inference_per_subject,
    load_subject_streamlines,
    compute_subject_wdice,
    plot_wdice_results,
)


def load_model(
    checkpoint_path: str,
    config: TrainConfig,
    device: torch.device
) -> nn.Module:
    """Load a trained model from checkpoint.
    
    Reads the saved 'params' dict from the checkpoint to reconstruct
    the exact model architecture used during training, falling back
    to the provided config for any missing keys.
    """
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Override config with params saved in the checkpoint (if available)
    saved_params = checkpoint.get('params', {})
    if saved_params:
        print(f"Loading model config from checkpoint params")
        for key in ['encoder_type', 'd_model', 'nhead', 'num_layers', 
                    'dim_feedforward', 'num_classes', 'dropout', 
                    'pooling', 'pos_encoding', 'input_size', 'norm_layer',
                    'deep_classifier']:
            if key in saved_params:
                setattr(config, key, saved_params[key])
                print(f"  {key}: {saved_params[key]}")
    else:
        print("WARNING: No 'params' found in checkpoint, using default config")
    
    # Create model based on config
    if config.encoder_type == 'transformer':
        model = TransformerEncoder(
            input_size=config.input_size,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            num_classes=config.num_classes,
            dropout=config.dropout,
            pooling=config.pooling,
            pos_encoding=config.pos_encoding,
            norm_layer=getattr(config, 'norm_layer', 'layernorm'),
            deep_classifier=getattr(config, 'deep_classifier', False)
        )
    else:
        model = LSTMEncoder(
            input_size=config.input_size,
            hidden_size=config.d_model,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
    
    # Handle state_dict keys from torch.compile (prefixed with '_orig_mod.')
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_accuracy' in checkpoint:
        print(f"Checkpoint validation accuracy: {checkpoint['val_accuracy']:.2f}%")
    if 'val_macro_f1' in checkpoint:
        print(f"Checkpoint validation F1: {checkpoint['val_macro_f1']:.2f}%")
    
    return model, checkpoint


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    preds_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on the test set, optionally loading/saving to preds_dir.
    
    Returns:
        all_labels: Ground truth labels
        all_preds: Predicted labels
        all_probs: Softmax probabilities for all classes
    """
    if preds_dir is not None:
        preds_file = os.path.join(preds_dir, "global_preds.npz")
        if os.path.exists(preds_file):
            print(f"📦 Loading saved global predictions from {preds_file}")
            data = np.load(preds_file)
            return data['labels'], data['preds'], data['probs']
            
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
    
    if preds_dir is not None:
        os.makedirs(preds_dir, exist_ok=True)
        preds_file = os.path.join(preds_dir, "global_preds.npz")
        np.savez_compressed(preds_file, labels=all_labels, preds=all_preds, probs=all_probs)
        print(f"💾 Saved global predictions to {preds_file}")
        
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
    all_classes = list(range(num_classes))
    metrics['precision_per_class'] = precision_score(labels, preds, labels=all_classes, average=None, zero_division=0) * 100
    metrics['recall_per_class'] = recall_score(labels, preds, labels=all_classes, average=None, zero_division=0) * 100
    metrics['f1_per_class'] = f1_score(labels, preds, labels=all_classes, average=None, zero_division=0) * 100
    
    # Top-k accuracy
    if num_classes > 2:
        for k in [3, 5]:
            if k < num_classes:
                metrics[f'top_{k}_accuracy'] = top_k_accuracy_score(
                    labels,
                    probs,
                    k=k,
                    labels=list(range(num_classes))
                ) * 100
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(labels, preds)
    
    # Classification report
    target_names = class_names if class_names else [f"Class {i}" for i in range(num_classes)]
    present_classes = np.unique(np.concatenate([labels, preds]))
    
    metrics['classification_report'] = classification_report(
        labels, 
        preds, 
        labels=present_classes,
        target_names=[target_names[i] for i in present_classes],
        zero_division=0
    )
    
    # Sample counts per class
    unique, counts = np.unique(labels, return_counts=True)
    metrics['samples_per_class'] = dict(zip(unique.tolist(), counts.tolist()))
    
    # Expected Calibration Error (ECE)
    n_bins = 15
    confidences = probs.max(axis=1)
    correct = (labels == preds).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences >= lo) & (confidences < hi) if hi < 1 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_counts.append(0)
            continue
        bin_accs.append(correct[mask].mean())
        bin_confs.append(confidences[mask].mean())
        bin_counts.append(int(mask.sum()))
    total = len(labels)
    ece = sum(bc * abs(ba - bf) for ba, bf, bc in zip(bin_accs, bin_confs, bin_counts)) / total
    metrics['ece'] = ece * 100  # as percentage
    metrics['ece_bin_accs'] = bin_accs
    metrics['ece_bin_confs'] = bin_confs
    metrics['ece_bin_counts'] = bin_counts
    metrics['ece_bin_boundaries'] = bin_boundaries.tolist()
    
    # ROC-AUC (One-vs-Rest)
    try:
        labels_bin = label_binarize(labels, classes=list(range(num_classes)))
        per_class_auc = []
        for i in range(num_classes):
            if labels_bin[:, i].sum() == 0:
                per_class_auc.append(float('nan')) # class never appears in the test set
                continue
            fpr_i, tpr_i, _ = roc_curve(labels_bin[:, i], probs[:, i])
            per_class_auc.append(auc(fpr_i, tpr_i))
        metrics['auc_per_class'] = per_class_auc
        valid_aucs = [a for a in per_class_auc if not np.isnan(a)]
        metrics['auc_macro'] = float(np.mean(valid_aucs)) if valid_aucs else 0.0
        
        valid_weights = [metrics['samples_per_class'].get(i, 0)
                         for i, a in enumerate(per_class_auc) if not np.isnan(a)]
                         
        metrics['auc_weighted'] = float(np.average(valid_aucs, weights=valid_weights)) if valid_aucs else 0.0
    except Exception as e:
        print(f"⚠ ROC-AUC computation failed: {e}")
        metrics['auc_per_class'] = []
        metrics['auc_macro'] = 0.0
        metrics['auc_weighted'] = 0.0
        
    # PR-AUC (Average Precision) One-vs-Rest
    try:
        from sklearn.metrics import average_precision_score
        if 'labels_bin' not in locals():
            labels_bin = label_binarize(labels, classes=list(range(num_classes)))
        per_class_pr_auc = []
        for i in range(num_classes):
            if labels_bin[:, i].sum() == 0:
                per_class_pr_auc.append(float('nan')) # class never appears in the test set
                continue
            ap = average_precision_score(labels_bin[:, i], probs[:, i])
            per_class_pr_auc.append(ap)
        metrics['pr_auc_per_class'] = per_class_pr_auc
        valid_pr_aucs = [a for a in per_class_pr_auc if not np.isnan(a)]
        metrics['pr_auc_macro'] = float(np.mean(valid_pr_aucs)) if valid_pr_aucs else 0.0
        
        valid_weights = [metrics['samples_per_class'].get(i, 0)
                         for i, a in enumerate(per_class_pr_auc) if not np.isnan(a)]
                         
        metrics['pr_auc_weighted'] = float(np.average(valid_pr_aucs, weights=valid_weights)) if valid_pr_aucs else 0.0
    except Exception as e:
        print(f"⚠ PR-AUC computation failed: {e}")
        metrics['pr_auc_per_class'] = []
        metrics['pr_auc_macro'] = 0.0
        metrics['pr_auc_weighted'] = 0.0
        
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
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(
            cm.astype('float'), 
            row_sums, 
            out=np.zeros_like(cm, dtype='float'), 
            where=row_sums != 0
        )
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


def plot_ece_diagram(
    metrics: Dict,
    save_path: str,
    figsize: Tuple[int, int] = (8, 6)
):
    """Plot reliability (calibration) diagram with ECE gap bars."""
    
    bin_accs = np.array(metrics['ece_bin_accs'])
    bin_confs = np.array(metrics['ece_bin_confs'])
    bin_counts = np.array(metrics['ece_bin_counts'])
    boundaries = np.array(metrics['ece_bin_boundaries'])
    bin_centers = (boundaries[:-1] + boundaries[1:]) / 2
    bin_widths = boundaries[1:] - boundaries[:-1]
    ece = metrics['ece']
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
    
    # Accuracy bars
    non_empty = bin_counts > 0
    ax1.bar(bin_centers[non_empty], bin_accs[non_empty], width=bin_widths[non_empty],
            alpha=0.6, color='#3498db', edgecolor='#2980b9', label='Accuracy')
    
    # Gap bars (|accuracy - confidence|)
    gaps = np.abs(bin_accs - bin_confs)
    ax1.bar(bin_centers[non_empty], gaps[non_empty], width=bin_widths[non_empty],
            bottom=np.minimum(bin_accs, bin_confs)[non_empty],
            alpha=0.35, color='#e74c3c', edgecolor='#c0392b', hatch='//',
            label=f'Gap (ECE = {ece:.2f}%)')
    
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Reliability Diagram (Expected Calibration Error)', fontsize=14)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Secondary axis: sample counts per bin
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, bin_counts, width=bin_widths, alpha=0.15,
            color='gray', edgecolor='none')
    ax2.set_ylabel('Samples per bin', fontsize=10, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved reliability diagram to {save_path}")


def plot_roc_auc(
    labels: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
    save_path: str,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """Plot One-vs-Rest ROC curves for all classes + macro average."""
    
    labels_bin = label_binarize(labels, classes=list(range(num_classes)))
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute per-class ROC
    all_fpr, all_tpr, all_auc = {}, {}, {}
    for i in range(num_classes):
        if labels_bin[:, i].sum() == 0:
            continue
        fpr_i, tpr_i, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc_i = auc(fpr_i, tpr_i)
        all_fpr[i] = fpr_i
        all_tpr[i] = tpr_i
        all_auc[i] = roc_auc_i
    
    # Plot individual class curves (semi-transparent)
    cmap = plt.cm.tab20(np.linspace(0, 1, num_classes))
    for i in sorted(all_auc.keys()):
        ax.plot(all_fpr[i], all_tpr[i], color=cmap[i], alpha=0.4, linewidth=0.8)
    
    # Macro-average ROC curve
    fpr_grid = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(fpr_grid)
    for i in all_tpr:
        mean_tpr += np.interp(fpr_grid, all_fpr[i], all_tpr[i])
    mean_tpr /= len(all_tpr)
    macro_auc = auc(fpr_grid, mean_tpr)
    
    ax.plot(fpr_grid, mean_tpr, color='navy', linewidth=2.5,
            label=f'Macro-avg ROC (AUC = {macro_auc:.4f})')
    
    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC-AUC plot to {save_path}")

def plot_pr_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
    save_path: str,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """Plot One-vs-Rest Precision-Recall curves for all classes + macro average."""
    
    from sklearn.metrics import precision_recall_curve, average_precision_score
    labels_bin = label_binarize(labels, classes=list(range(num_classes)))
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    all_precision, all_recall, all_ap = {}, {}, {}
    for i in range(num_classes):
        if labels_bin[:, i].sum() == 0:
            continue
        precision_i, recall_i, _ = precision_recall_curve(labels_bin[:, i], probs[:, i])
        ap_i = average_precision_score(labels_bin[:, i], probs[:, i])
        all_precision[i] = precision_i
        all_recall[i] = recall_i
        all_ap[i] = ap_i
    
    cmap = plt.cm.tab20(np.linspace(0, 1, num_classes))
    for i in sorted(all_ap.keys()):
        ax.plot(all_recall[i], all_precision[i], color=cmap[i], alpha=0.4, linewidth=0.8)
    
    # Calculate Macro-average PR curve
    try:
        recall_grid = np.linspace(0, 1, 200)
        mean_precision = np.zeros_like(recall_grid)
        valid_classes = 0
        for i in all_recall:
            # precision_recall_curve returns recall from high to low, so reverse it
            rec = all_recall[i][::-1]
            prec = all_precision[i][::-1]
            mean_precision += np.interp(recall_grid, rec, prec)
            valid_classes += 1
        
        if valid_classes > 0:
            mean_precision /= valid_classes
            macro_ap = np.mean([all_ap[i] for i in all_ap])
            ax.plot(recall_grid, mean_precision, color='navy', linewidth=2.5,
                    label=f'Macro-avg PR (AP = {macro_ap:.4f})')
    except Exception as e:
        print(f"Could not compute macro-average PR curve: {e}")
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves (One-vs-Rest)', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curve plot to {save_path}")


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
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=cfg.num_workers,
                        help='DataLoader workers')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision')
    parser.add_argument('--sampling_pct_test', type=float, default=0.25,
                        help='Percentage of test data to use (1.0 = all)')
    parser.add_argument('--pos_encoding', type=str, default=cfg.pos_encoding,
                        choices=['absolute', 'rope'],
                        help='Positional encoding type (must match trained model)')
    
    # ── wDice arguments ──
    parser.add_argument('--wdice', action='store_true',
                        help='Also compute volumetric weighted Dice (wDice) after classification metrics')
    parser.add_argument('--tractoinferno_dir', type=str,
                        default='/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives',
                        help='Path to Tractoinferno derivatives directory (for wDice)')
    parser.add_argument('--scope', type=str, default='testset',
                        choices=['trainset', 'validset', 'testset'],
                        help='Dataset scope for wDice evaluation')
    parser.add_argument('--preds_dir', type=str, default=None,
                        help='Directory to save/load inference predictions to avoid recomputing')
    parser.add_argument('--bootstrap', action='store_true',
                        help='Compute 95%% Confidence Intervals for Accuracy, wDice, Dice using subject-level bootstrapping')
    parser.add_argument('--n_bootstraps', type=int, default=1000,
                        help='Number of bootstrap resamples (default: 1000)')

    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    cfg.pos_encoding = args.pos_encoding
    model, checkpoint = load_model(args.checkpoint, cfg, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get test files
    test_dir = Path(args.test_dir)
    test_files = sorted([str(f) for f in test_dir.glob('*.hdf5')])
    print(f"\nFound {len(test_files)} test HDF5 files in {test_dir}")
    
    if not test_files:
        print("ERROR: No HDF5 files found in test directory!")
        return
    
    import time
    start_time = time.time()
    
    subject_predictions = {}
    
    if args.wdice:
        print("\n" + "=" * 60)
        print("RUNNING INFERENCE PER SUBJECT (for Metrics & wDice)")
        print("=" * 60)
        
        all_labels_list = []
        all_preds_list = []
        all_probs_list = []
        
        for i, hdf5_path in enumerate(test_files):
            subject_name = Path(hdf5_path).stem
            print(f"  [{i+1}/{len(test_files)}] Inferring {subject_name}...")
            gt_labels_subj, pred_labels_subj, probs_subj = run_inference_per_subject(
                model, str(hdf5_path), device,
                batch_size=args.batch_size,
                use_amp=not args.no_amp,
                preds_dir=args.preds_dir
            )
            subject_predictions[subject_name] = (gt_labels_subj, pred_labels_subj, probs_subj)
            
            all_labels_list.extend(gt_labels_subj)
            all_preds_list.extend(pred_labels_subj)
            all_probs_list.append(probs_subj)
            
        labels = np.array(all_labels_list)
        preds = np.array(all_preds_list)
        probs = np.concatenate(all_probs_list, axis=0) if all_probs_list else np.array([])
    else:
        # Create test dataset (use all data by default for testing)
        test_dataset = StreamlineDataset(
            test_files,
            sampling_percentage=args.sampling_pct_test,
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
        labels, preds, probs = evaluate_model(model, test_loader, device, use_amp=not args.no_amp, preds_dir=args.preds_dir)

    inference_time = time.time() - start_time
    print(f"\n⏱️ Inference completed in {inference_time:.2f} seconds.")
    
    # Get class names from ID_TO_TRACT mapping
    unique_classes = np.unique(np.concatenate([labels, preds]))
    class_names = [ID_TO_TRACT.get(i, f"Bundle_{i}") for i in range(cfg.num_classes)]
    
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
    
    print(f"   ECE:               {metrics['ece']:.2f}%")
    print(f"   ROC-AUC (macro):   {metrics['auc_macro']:.4f}")
    print(f"   ROC-AUC (weighted):{metrics['auc_weighted']:.4f}")
    print(f"   PR-AUC (macro):    {metrics['pr_auc_macro']:.4f}")
    print(f"   PR-AUC (weighted): {metrics['pr_auc_weighted']:.4f}")
    
    print(f"\n📋 Classification Report:\n")
    print(metrics['classification_report'])
    
    # Save classification report to file
    os.makedirs(args.output_dir, exist_ok=True)  # Safeguard in case dir was deleted during inference
    report_path = os.path.join(args.output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(metrics['classification_report'])
    print(f"💾 Classification report saved to {report_path}")
    
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
    
    # 7. Reliability diagram (ECE)
    plot_ece_diagram(
        metrics,
        os.path.join(args.output_dir, 'reliability_diagram.png')
    )
    
    # 8. ROC-AUC curves
    plot_roc_auc(
        labels, probs, cfg.num_classes,
        os.path.join(args.output_dir, 'roc_auc.png'),
        class_names=class_names
    )
    
    # 9. Precision-Recall curves
    plot_pr_curve(
        labels, probs, cfg.num_classes,
        os.path.join(args.output_dir, 'pr_auc.png'),
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
        'total_incorrect': int((labels != preds).sum()),
        'inference_time_seconds': inference_time
    }
    
    if 'top_3_accuracy' in metrics:
        metrics_to_save['top_3_accuracy'] = metrics['top_3_accuracy']
    if 'top_5_accuracy' in metrics:
        metrics_to_save['top_5_accuracy'] = metrics['top_5_accuracy']
    
    metrics_to_save['ece'] = metrics['ece']
    metrics_to_save['auc_macro'] = metrics['auc_macro']
    metrics_to_save['auc_weighted'] = metrics['auc_weighted']
    metrics_to_save['pr_auc_macro'] = metrics['pr_auc_macro']
    metrics_to_save['pr_auc_weighted'] = metrics['pr_auc_weighted']
    
    metrics_file = os.path.join(args.output_dir, 'test_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nSaved metrics to {metrics_file}")
    
    # ── Optional wDice evaluation ──
    if args.wdice:
        print("\n" + "=" * 60)
        print("WEIGHTED DICE (wDice) EVALUATION")
        print("=" * 60)

        from utils.dataset_handler import Tractoinferno_handler
        from collections import defaultdict as _defaultdict

        wdice_out = os.path.join(args.output_dir, 'wDice')
        os.makedirs(wdice_out, exist_ok=True)

        # Get test HDF5 files
        hdf5_files = sorted(test_dir.glob('*.hdf5'))

        # Set up dataset handler for loading original .trk files
        dataset_handler = Tractoinferno_handler(args.tractoinferno_dir, scope=args.scope)
        subjects_data = {s['subject']: s for s in dataset_handler.get_data()}

        # Process each subject
        all_subject_results = {}
        aggregate_wdice = _defaultdict(list)

        for i, hdf5_path in enumerate(hdf5_files):
            subject_name = hdf5_path.stem
            print(f"\n{'─'*60}")
            print(f"[{i+1}/{len(hdf5_files)}] Processing {subject_name}")
            print(f"{'─'*60}")

            if subject_name not in subjects_data:
                print(f"  ⚠ Subject {subject_name} not found in Tractoinferno {args.scope}, skipping")
                continue

            subject_info = subjects_data[subject_name]
            subject_path = pathlib.Path(subject_info['T1w']).parent.parent
            mri_path = subject_info['T1w']

            # Retrieve already computed per-subject inference
    
            gt_labels_subj, pred_labels_subj, _ = subject_predictions[subject_name]
            accuracy_subj = np.mean(np.array(gt_labels_subj) == np.array(pred_labels_subj)) * 100
            print(f"  📈 Subject accuracy: {accuracy_subj:.2f}%")

            # Load original .trk streamlines
            print(f"  📂 Loading original .trk streamlines...")
            tract_streamlines, affine = load_subject_streamlines(subject_path, mri_path)
            total_original = sum(len(sls) for sls in tract_streamlines.values())
            print(f"  📊 Loaded {total_original} original streamlines from {len(tract_streamlines)} bundles")

            # Compute wDice
            print(f"  🎲 Voxelizing and computing wDice...")
            subject_results = compute_subject_wdice(
                tract_streamlines, gt_labels_subj, pred_labels_subj, affine
            )
            subject_results['accuracy'] = float(accuracy_subj)

            all_subject_results[subject_name] = subject_results
            for bundle, score in subject_results.items():
                aggregate_wdice[bundle].append(score)

            print(f"  ✅ Mean wDice: {subject_results['mean_wDice']:.4f}  |  Mean Dice: {subject_results.get('mean_Dice', 0.0):.4f}")

        # Aggregate results across subjects
        if all_subject_results:
            print(f"\n{'='*60}")
            print("AGGREGATE wDice/Dice RESULTS")
            print(f"{'='*60}")

            mean_results = {}
            for bundle, scores in aggregate_wdice.items():
                if bundle not in ['mean_wDice', 'mean_Dice']:
                    mean_results[bundle] = float(np.mean(scores))
                    
            wDice_keys = [k for k in mean_results.keys() if not k.endswith('_Dice')]
            Dice_keys = [k for k in mean_results.keys() if k.endswith('_Dice')]

            mean_results['mean_wDice'] = float(np.mean([mean_results[k] for k in wDice_keys])) if wDice_keys else 0.0
            mean_results['mean_Dice'] = float(np.mean([mean_results[k] for k in Dice_keys])) if Dice_keys else 0.0

            print(f"\nOverall Mean wDice: {mean_results['mean_wDice']:.4f}")
            print(f"Overall Mean Dice:  {mean_results['mean_Dice']:.4f}")
            
            print(f"\nPer-bundle wDice (averaged across {len(all_subject_results)} subjects):")
            for bundle in sorted(wDice_keys):
                print(f"  {bundle:15s}: {mean_results[bundle]:.4f}")

            print(f"\nPer-bundle Dice (averaged across {len(all_subject_results)} subjects):")
            for bundle in sorted(Dice_keys):
                base_name = bundle.replace('_Dice', '')
                print(f"  {base_name:15s}: {mean_results[bundle]:.4f}")

            # Save per-subject JSON
            json_path = os.path.join(wdice_out, 'wdice_per_subject.json')
            with open(json_path, 'w') as f:
                json.dump(all_subject_results, f, indent=2)
            print(f"\n💾 Per-subject metrics saved to {json_path}")

            # Save aggregate JSON
            agg_json_path = os.path.join(wdice_out, 'wdice_aggregate.json')
            with open(agg_json_path, 'w') as f:
                json.dump(mean_results, f, indent=2)
            print(f"💾 Aggregate metrics saved to {agg_json_path}")

            # CSV summary
            try:
                import pandas as pd
                rows = []
                for subj, results in all_subject_results.items():
                    row = {'subject': subj}
                    row.update(results)
                    rows.append(row)
                df = pd.DataFrame(rows)
                csv_path = os.path.join(wdice_out, 'wdice_results.csv')
                df.to_csv(csv_path, index=False)
                print(f"💾 CSV summary saved to {csv_path}")
            except ImportError:
                pass

            # Plot wDice results
            wdice_plot_data = {k: v for k, v in mean_results.items() if not k.endswith('_Dice')}
            plot_wdice_results(
                wdice_plot_data,
                os.path.join(wdice_out, 'wdice_per_bundle.png'),
                title="Per-Bundle wDice (Averaged Across Subjects)"
            )
            
            dice_plot_data = {k.replace('_Dice', ''): v for k, v in mean_results.items() if k.endswith('_Dice') or k == 'mean_Dice'}
            if 'mean_Dice' in mean_results:
                dice_plot_data['mean_wDice'] = mean_results['mean_Dice'] # rename for compat
                
            plot_wdice_results(
                dice_plot_data,
                os.path.join(wdice_out, 'dice_per_bundle.png'),
                title="Per-Bundle Dice (Averaged Across Subjects)"
            )

            # Add wDice and Dice to the saved metrics JSON
            metrics_to_save['mean_wDice'] = mean_results['mean_wDice']
            metrics_to_save['mean_Dice'] = mean_results['mean_Dice']
            with open(metrics_file, 'w') as f:
                json.dump(metrics_to_save, f, indent=2)
            print(f"📝 Updated {metrics_file} with mean_wDice and mean_Dice")
            
            # --- Bootstrap subject-level metrics ---
            if args.bootstrap and len(all_subject_results) > 1:
                print(f"\n{'='*60}")
                print(f"BOOTSTRAP 95% CONFIDENCE INTERVALS (N={args.n_bootstraps})")
                print(f"{'='*60}")
                rng = np.random.default_rng(42)
                subjs = list(all_subject_results.keys())
                n_subjs = len(subjs)
                
                bs_wdice, bs_dice, bs_acc = [], [], []
                
                print("Computing CIs via subject-level resampling...")
                for _ in range(args.n_bootstraps):
                    indices = rng.choice(n_subjs, size=n_subjs, replace=True)
                    sampled = [subjs[idx] for idx in indices]
                    bs_wdice.append(np.mean([all_subject_results[s].get('mean_wDice', 0.0) for s in sampled]))
                    bs_dice.append(np.mean([all_subject_results[s].get('mean_Dice', 0.0) for s in sampled]))
                    bs_acc.append(np.mean([all_subject_results[s].get('accuracy', 0.0) for s in sampled]))
                
                print(f"  Subject-level Accuracy: {np.mean(bs_acc):.2f}% (95% CI: [{np.percentile(bs_acc, 2.5):.2f}%, {np.percentile(bs_acc, 97.5):.2f}%])")
                print(f"  Subject-level wDice:    {np.mean(bs_wdice):.4f}  (95% CI: [{np.percentile(bs_wdice, 2.5):.4f}, {np.percentile(bs_wdice, 97.5):.4f}])")
                print(f"  Subject-level Dice:     {np.mean(bs_dice):.4f}  (95% CI: [{np.percentile(bs_dice, 2.5):.4f}, {np.percentile(bs_dice, 97.5):.4f}])")
                
                metrics_to_save['bootstrap_95ci_accuracy'] = [np.percentile(bs_acc, 2.5), np.percentile(bs_acc, 97.5)]
                metrics_to_save['bootstrap_95ci_wDice'] = [np.percentile(bs_wdice, 2.5), np.percentile(bs_wdice, 97.5)]
                metrics_to_save['bootstrap_95ci_Dice'] = [np.percentile(bs_dice, 2.5), np.percentile(bs_dice, 97.5)]
                
                with open(metrics_file, 'w') as f:
                    json.dump(metrics_to_save, f, indent=2)
                print(f"📝 Appended Bootstrap CIs to {metrics_file}")

        else:
            print("\n⚠ No subjects processed for wDice!")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\n📁 All results saved to: {args.output_dir}/")
    if args.wdice:
        print(f"📁 wDice results saved to: {wdice_out}/")


def compare_test_results(results_dir: str = 'tests/test_results_2',
                         wdice_dir: str = 'tests/wdice_results'):
    """
    Compare test results across all experiments.

    Scans subdirectories of results_dir for test_metrics.json and
    classification_report.txt, then generates:
      - summary_comparison.csv  (overall metrics per experiment, including mean_wDice)
      - per_class_comparison.csv (per-class F1 with best/worst analysis)
      - overview_comparison.png
      - per_class_f1_heatmap.png
      - per_class_f1_delta.png
      - per_class_f1_scatter.png
      - radar_comparison.png
      - best_model_per_class.png
    """
    import pandas as pd
    import re
    from matplotlib.patches import FancyBboxPatch

    results_path = Path(results_dir)
    comparison_dir = results_path / 'comparison'
    comparison_dir.mkdir(exist_ok=True)

    # --- 1. Collect metrics from every experiment ---
    experiments = {}
    per_class_f1 = {}

    for exp_dir in sorted(results_path.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == 'comparison':
            continue
        metrics_file = exp_dir / 'test_metrics.json'
        report_file = exp_dir / 'classification_report.txt'
        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        # Look for wDice data if not already in test_metrics.json
        if 'mean_wDice' not in metrics:
            wdice_path = Path(wdice_dir)
            if wdice_path.is_dir():
                # Try exact match first, then partial/substring match
                wdice_agg = wdice_path / exp_dir.name / 'wdice_aggregate.json'
                if not wdice_agg.exists():
                    # Try matching: wdice dir name is a substring of experiment name
                    for wd in sorted(wdice_path.iterdir()):
                        if wd.is_dir() and wd.name in exp_dir.name:
                            candidate = wd / 'wdice_aggregate.json'
                            if candidate.exists():
                                wdice_agg = candidate
                                break
                if wdice_agg.exists():
                    with open(wdice_agg) as wf:
                        wdice_data = json.load(wf)
                    if 'mean_wDice' in wdice_data:
                        metrics['mean_wDice'] = wdice_data['mean_wDice']
                    if 'mean_Dice' in wdice_data:
                        metrics['mean_Dice'] = wdice_data['mean_Dice']

        experiments[exp_dir.name] = metrics

        # Parse per-class F1 from classification_report.txt
        if report_file.exists():
            with open(report_file) as f:
                lines = f.readlines()
            class_f1 = {}
            for line in lines:
                # Match lines like "    Bundle_0       0.99      1.00      1.00    279842"
                m = re.match(r'\s+(Bundle_\d+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+\d+', line)
                if m:
                    class_f1[m.group(1)] = float(m.group(2))
            per_class_f1[exp_dir.name] = class_f1

    if not experiments:
        print("No experiment results found!")
        return

    print(f"Found {len(experiments)} experiments to compare")

    # --- 2. Summary comparison CSV ---
    summary_rows = []
    metric_keys = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro',
                   'recall_macro', 'top_3_accuracy', 'top_5_accuracy',
                   'ece', 'auc_macro', 'auc_weighted', 'mean_wDice', 'mean_Dice']
    for name, m in experiments.items():
        row = {'experiment': name}
        for k in metric_keys:
            row[k] = m.get(k, None)
        # Convert mean_wDice and mean_Dice to percentage for consistency with other metrics
        if row.get('mean_wDice') is not None:
            row['mean_wDice'] = row['mean_wDice'] * 100
        if row.get('mean_Dice') is not None:
            row['mean_Dice'] = row['mean_Dice'] * 100
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows).sort_values('f1_macro', ascending=False)
    df_summary.to_csv(comparison_dir / 'summary_comparison.csv', index=False)
    print(f"  Saved summary_comparison.csv")

    # --- 3. Per-class comparison CSV ---
    if per_class_f1:
        df_class = pd.DataFrame(per_class_f1)
        # Add analysis columns
        df_class['best_model'] = df_class.idxmax(axis=1)
        df_class['best_f1'] = df_class.drop(columns=['best_model'], errors='ignore').max(axis=1)
        df_class['worst_f1'] = df_class.drop(columns=['best_model', 'best_f1'], errors='ignore').min(axis=1)
        df_class['delta'] = df_class['best_f1'] - df_class['worst_f1']
        df_class.to_csv(comparison_dir / 'per_class_comparison.csv')
        print(f"  Saved per_class_comparison.csv")

    # --- 4. Overview comparison plot ---
    _plot_overview_comparison(df_summary, comparison_dir)

    # --- 5. Per-class F1 heatmap ---
    if per_class_f1:
        _plot_per_class_heatmap(per_class_f1, comparison_dir)
        _plot_per_class_delta(per_class_f1, comparison_dir)
        _plot_per_class_scatter(per_class_f1, comparison_dir)
        _plot_best_model_per_class(per_class_f1, comparison_dir)

    # --- 6. Radar comparison ---
    _plot_radar_comparison(experiments, comparison_dir)

    print(f"\n📁 Comparison results saved to: {comparison_dir}/")


def _plot_overview_comparison(df_summary: 'pd.DataFrame', out_dir: Path):
    """Bar chart comparing overall metrics across experiments."""
    import pandas as pd

    metrics_to_plot = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    available = [m for m in metrics_to_plot if m in df_summary.columns]

    fig, ax = plt.subplots(figsize=(max(14, len(df_summary) * 1.5), 7))
    x = np.arange(len(df_summary))
    width = 0.15
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

    for i, metric in enumerate(available):
        offset = (i - len(available) / 2) * width
        vals = df_summary[metric].values
        ax.bar(x + offset, vals, width, label=metric.replace('_', ' ').title(), color=colors[i % len(colors)])

    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Overall Metrics Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df_summary['experiment'], rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(min(df_summary[available].min().min() - 2, 85), 100)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'overview_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved overview_comparison.png")


def _plot_per_class_heatmap(per_class_f1: Dict, out_dir: Path):
    """Heatmap of per-class F1 across experiments."""
    import pandas as pd

    df = pd.DataFrame(per_class_f1)

    fig, ax = plt.subplots(figsize=(max(12, len(df.columns) * 1.2), max(10, len(df) * 0.4)))
    sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
                vmin=0.4, vmax=1.0, linewidths=0.5,
                cbar_kws={'label': 'F1 Score'})
    ax.set_title('Per-Class F1 Score Comparison', fontsize=14)
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Class', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / 'per_class_f1_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved per_class_f1_heatmap.png")


def _plot_per_class_delta(per_class_f1: Dict, out_dir: Path):
    """Bar chart showing F1 variability (max - min) per class across experiments."""
    import pandas as pd

    df = pd.DataFrame(per_class_f1)
    deltas = df.max(axis=1) - df.min(axis=1)
    deltas = deltas.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(max(12, len(deltas) * 0.5), 6))
    colors = ['#e74c3c' if d > 0.03 else '#f39c12' if d > 0.01 else '#2ecc71' for d in deltas.values]
    ax.bar(range(len(deltas)), deltas.values, color=colors, edgecolor='#2c3e50', linewidth=0.5)
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels(deltas.index, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('F1 Delta (max - min)', fontsize=12)
    ax.set_title('Per-Class F1 Variability Across Experiments', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (idx, val) in enumerate(deltas.items()):
        ax.text(i, val + 0.002, f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / 'per_class_f1_delta.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved per_class_f1_delta.png")


def _plot_per_class_scatter(per_class_f1: Dict, out_dir: Path):
    """Scatter plot: each dot is (class, F1) with one series per experiment."""
    import pandas as pd

    df = pd.DataFrame(per_class_f1)
    exp_names = df.columns.tolist()

    fig, ax = plt.subplots(figsize=(max(14, len(df) * 0.5), 7))
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_names)))

    for i, exp in enumerate(exp_names):
        ax.scatter(range(len(df)), df[exp].values,
                   label=exp, marker=markers[i % len(markers)],
                   color=colors[i], s=50, alpha=0.8, edgecolors='#2c3e50', linewidths=0.5)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Scores Across Experiments', fontsize=14)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.05)
    plt.tight_layout()
    plt.savefig(out_dir / 'per_class_f1_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved per_class_f1_scatter.png")


def _plot_best_model_per_class(per_class_f1: Dict, out_dir: Path):
    """Horizontal bar chart showing which model is best for each class."""
    import pandas as pd

    df = pd.DataFrame(per_class_f1)
    best_models = df.idxmax(axis=1)
    best_f1 = df.max(axis=1)

    unique_models = best_models.unique()
    model_colors = {m: plt.cm.Set2(i / max(len(unique_models) - 1, 1))
                    for i, m in enumerate(unique_models)}

    fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.35)))
    y_pos = np.arange(len(df))
    bar_colors = [model_colors[m] for m in best_models.values]

    ax.barh(y_pos, best_f1.values, color=bar_colors, edgecolor='#2c3e50', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index, rotation=0, ha='right', fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Best F1 Score', fontsize=12)
    ax.set_title('Best Model Per Class', fontsize=14)
    ax.set_xlim(0.4, 1.05)
    ax.grid(axis='x', alpha=0.3)

    # Add model name labels
    for i, (f1_val, model) in enumerate(zip(best_f1.values, best_models.values)):
        ax.text(f1_val + 0.005, i, f'{model} ({f1_val:.2f})',
                va='center', fontsize=8)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=model_colors[m]) for m in unique_models]
    ax.legend(handles, unique_models, fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.savefig(out_dir / 'best_model_per_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved best_model_per_class.png")


def _plot_radar_comparison(experiments: Dict, out_dir: Path):
    """Radar chart comparing key metrics across experiments."""

    metrics_for_radar = ['accuracy', 'f1_macro', 'f1_weighted',
                         'precision_macro', 'recall_macro']
    labels = [m.replace('_', ' ').title() for m in metrics_for_radar]
    num_vars = len(labels)

    # Compute angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    for i, (name, metrics) in enumerate(experiments.items()):
        values = [metrics.get(m, 0) for m in metrics_for_radar]
        values += values[:1]  # Close the polygon
        ax.plot(angles, values, 'o-', linewidth=1.5, label=name,
                color=colors[i], markersize=4)
        ax.fill(angles, values, alpha=0.05, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(85, 95)
    ax.set_title('Metrics Radar Comparison', fontsize=14, pad=20)
    ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'radar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved radar_comparison.png")


if __name__ == '__main__':
    import sys as _sys
    if '--compare' in _sys.argv:
        # Remove --compare from argv so argparse doesn't complain
        _sys.argv.remove('--compare')
        # Check for optional --results_dir argument
        results_dir = 'tests/test_results_2'
        if '--results_dir' in _sys.argv:
            idx = _sys.argv.index('--results_dir')
            results_dir = _sys.argv[idx + 1]
            _sys.argv.pop(idx)
            _sys.argv.pop(idx)
        # Check for optional --wdice_dir argument
        wdice_dir = 'tests/wdice_results'
        if '--wdice_dir' in _sys.argv:
            idx = _sys.argv.index('--wdice_dir')
            wdice_dir = _sys.argv[idx + 1]
            _sys.argv.pop(idx)
            _sys.argv.pop(idx)
        compare_test_results(results_dir, wdice_dir)
    else:
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
python test.py --sampling_pct_test 0.1

# Run classification metrics + wDice in one go
python test.py --checkpoint checkpoints/exp/best_model.pt --wdice

# Run with custom wDice output directory
python test.py --wdice --wdice_output_dir tests/wdice_results/my_exp

# Compare all test results across experiments
python test.py --compare

# Compare results from a custom directory
python test.py --compare --results_dir path/to/test_results
"""