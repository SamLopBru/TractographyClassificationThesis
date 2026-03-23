"""
Weighted Dice (wDice) evaluation for tractography bundle classification.

Computes the volumetric weighted Dice score (Cousineau et al., 2017) by:
  1. Running model inference on spherical-encoded HDF5 data
  2. Loading original (x,y,z) streamlines from .trk files
  3. Voxelizing both reference and predicted bundles
  4. Computing per-bundle wDice on density maps

Usage:
    python tests/wDice.py --checkpoint checkpoints/<exp>/best_model.pt
    python tests/wDice.py --checkpoint checkpoints/<exp>/best_model.pt --scope testset
"""

import argparse
import json
import os
import sys
import pathlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import h5py
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import TrainConfig, DEFAULT_CONFIG
from src.encoder import TransformerEncoder, LSTMEncoder
from utils.dataset_handler import Tractoinferno_handler

# Tract name ↔ ID mapping (must match sequencer.py)
ENCODED_TRACTS = {
    'AF_L': 0, 'AF_R': 1, 'CC_Fr_1': 2, 'CC_Fr_2': 3, 'CC_Oc': 4,
    'CC_Pa': 5, 'CC_Pr_Po': 6, 'CG_L': 7, 'CG_R': 8, 'FAT_L': 9,
    'FAT_R': 10, 'FPT_L': 11, 'FPT_R': 12, 'FX_L': 13, 'FX_R': 14,
    'IFOF_L': 15, 'IFOF_R': 16, 'ILF_L': 17, 'ILF_R': 18, 'MCP': 19,
    'MdLF_L': 20, 'MdLF_R': 21, 'OR_ML_L': 22, 'OR_ML_R': 23, 'POPT_L': 24,
    'POPT_R': 25, 'PYT_L': 26, 'PYT_R': 27, 'SLF_L': 28, 'SLF_R': 29,
    'UF_L': 30, 'UF_R': 31
}
ID_TO_TRACT = {v: k for k, v in ENCODED_TRACTS.items()}


# ─────────────────────────────── Model loading ────────────────────────────── #

def load_model(
    checkpoint_path: str,
    config: TrainConfig,
    device: torch.device
) -> Tuple[nn.Module, dict]:
    """Load a trained model from checkpoint (reused from test.py)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    saved_params = checkpoint.get('params', {})
    if saved_params:
        for key in ['encoder_type', 'd_model', 'nhead', 'num_layers',
                     'dim_feedforward', 'num_classes', 'dropout',
                     'pooling', 'pos_encoding', 'input_size', 'norm_layer',
                     'deep_classifier']:
            if key in saved_params:
                setattr(config, key, saved_params[key])

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

    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, checkpoint


# ─────────────────────────── Voxelization & wDice ─────────────────────────── #

def voxelize_streamlines(
    streamlines: List[np.ndarray],
    affine: np.ndarray
) -> Dict[tuple, int]:
    """
    Convert a list of (x,y,z) streamlines into a voxel density map.

    Each voxel is counted once per streamline that passes through it.

    Args:
        streamlines: List of Nx3 arrays with world-space coordinates.
        affine: 4x4 NIfTI affine matrix (world → voxel via its inverse).

    Returns:
        Dictionary mapping (i,j,k) voxel tuples to streamline counts.
    """
    inv_affine = np.linalg.inv(affine)
    density: Dict[tuple, int] = defaultdict(int)

    for sl in streamlines:
        # Transform to voxel coordinates: [i,j,k,1] = A^-1 · [x,y,z,1]
        coords_h = np.hstack([sl, np.ones((len(sl), 1))])
        voxels = (inv_affine @ coords_h.T).T[:, :3]
        voxels = np.round(voxels).astype(np.int32)

        # Each voxel counted at most once per streamline
        unique_voxels = set(map(tuple, voxels))
        for v in unique_voxels:
            density[v] += 1

    return dict(density)


def compute_wdice(
    density_ref: Dict[tuple, int],
    density_pred: Dict[tuple, int]
) -> float:
    """
    Compute weighted Dice (Cousineau et al., 2017).

    wDice = 2 · Σ_{v'} min(W_ref,v', W_pred,v') / (Σ_v W_ref,v + Σ_v W_pred,v)

    where v' are voxels with positive density in both maps.
    """
    if not density_ref or not density_pred:
        return 0.0

    common_voxels = set(density_ref.keys()) & set(density_pred.keys())
    numerator = 2.0 * sum(
        min(density_ref[v], density_pred[v]) for v in common_voxels
    )
    denominator = sum(density_ref.values()) + sum(density_pred.values())

    return numerator / denominator if denominator > 0 else 0.0


def compute_dice(
    density_ref: Dict[tuple, int],
    density_pred: Dict[tuple, int]
) -> float:
    """
    Compute standard volumetric Dice score.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    where A and B are the sets of voxels with positive density.
    """
    if not density_ref or not density_pred:
        return 0.0

    common_voxels = set(density_ref.keys()) & set(density_pred.keys())
    numerator = 2.0 * len(common_voxels)
    denominator = len(density_ref) + len(density_pred)

    return numerator / denominator if denominator > 0 else 0.0


# ───────────────────────── Per-subject inference ──────────────────────────── #

@torch.no_grad()
def run_inference_per_subject(
    model: nn.Module,
    hdf5_path: str,
    device: torch.device,
    batch_size: int = 4096,
    use_amp: bool = True,
    preds_dir: Optional[str] = None
) -> Tuple[List[int], List[np.ndarray], List[np.ndarray]]:
    """
    Run inference on a single subject's HDF5 file.

    Iterates over tract groups in the HDF5 in sorted order (tract_0,
    tract_1, …) so that we can match predictions back to the original
    .trk files. If preds_dir is provided, saves or loads predictions to avoid
    recomputing.
    """
    subject_name = pathlib.Path(hdf5_path).stem
    if preds_dir is not None:
        preds_file = os.path.join(preds_dir, f"{subject_name}_preds.npz")
        if os.path.exists(preds_file):
            print(f"    📦 Loading saved predictions from {preds_file}")
            data = np.load(preds_file)
            return data['gt_labels'].tolist(), data['all_preds'].tolist(), data['all_probs']

    gt_labels = []
    all_preds = []
    all_probs = []

    with h5py.File(hdf5_path, 'r') as f:
        # Iterate groups in deterministic order (tract_0, tract_1, …)
        group_names = sorted(
            [g for g in f.keys() if g.startswith('tract_')],
            key=lambda g: int(g.split('_')[1])
        )

        for group_name in group_names:
            tract_group = f[group_name]
            tract_id = int(tract_group.attrs['tract_id'])
            n_streamlines = int(tract_group.attrs['n_streamlines'])
            lengths = tract_group['lengths'][:]
            data = tract_group['streamlines'][:]  # (N, max_len, 5)

            # Process in batches
            for start in range(0, n_streamlines, batch_size):
                end = min(start + batch_size, n_streamlines)
                batch_data = data[start:end]
                batch_lengths = lengths[start:end]

                # Trim to max length in this batch for efficiency
                max_len = int(batch_lengths.max())
                batch_data = batch_data[:, :max_len, :]

                streamlines_t = torch.from_numpy(
                    batch_data.astype(np.float32)
                ).to(device)
                lengths_t = torch.from_numpy(
                    batch_lengths.astype(np.int64)
                ).to(device)

                if use_amp:
                    with autocast(
                        device_type='cuda' if device.type == 'cuda' else 'cpu',
                        dtype=torch.float16
                    ):
                        logits = model(streamlines_t, lengths=lengths_t)
                else:
                    logits = model(streamlines_t, lengths=lengths_t)

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()

                gt_labels.extend([tract_id] * (end - start))
                all_preds.extend(preds.tolist())
                all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
    
    if preds_dir is not None:
        os.makedirs(preds_dir, exist_ok=True)
        preds_file = os.path.join(preds_dir, f"{subject_name}_preds.npz")
        np.savez_compressed(preds_file, 
                 gt_labels=np.array(gt_labels), 
                 all_preds=np.array(all_preds), 
                 all_probs=all_probs)
        print(f"    💾 Saved predictions to {preds_file}")
        
    return gt_labels, all_preds, all_probs


# ─────────────────── Load original .trk streamlines ──────────────────────── #

def load_subject_streamlines(
    subject_path: pathlib.Path,
    mri_path: pathlib.Path,
) -> Tuple[Dict[int, List[np.ndarray]], np.ndarray]:
    """
    Load original (x,y,z) streamlines from .trk files for one subject.

    Returns:
        tract_streamlines: Dict mapping tract_id to list of Nx3 arrays.
        affine: 4x4 NIfTI affine matrix from the subject's T1w image.
    """
    from dipy.io.streamline import load_tractogram

    mri = nib.load(str(mri_path))
    affine = mri.affine

    tract_streamlines: Dict[int, List[np.ndarray]] = {}
    tract_dir = subject_path / "tractography"

    for trk_file in sorted(tract_dir.glob("*.trk")):
        tract_name = trk_file.name.split("__")[-1].split(".")[0]
        if tract_name not in ENCODED_TRACTS:
            continue
        tract_id = ENCODED_TRACTS[tract_name]

        tractogram = load_tractogram(str(trk_file), str(mri_path))
        tract_streamlines[tract_id] = [
            sl.astype(np.float64) for sl in tractogram.streamlines
        ]
        del tractogram

    return tract_streamlines, affine


# ────────────────────── wDice for a single subject ────────────────────────── #

def compute_subject_wdice(
    tract_streamlines: Dict[int, List[np.ndarray]],
    gt_labels: List[int],
    pred_labels: List[int],
    affine: np.ndarray,
) -> Dict[str, float]:
    """
    Compute per-bundle wDice for one subject.

    The streamline ordering in gt_labels / pred_labels matches the HDF5
    group ordering (tract_0, tract_1, …), and within each group, matches
    the sequential order from the .trk file.

    Args:
        tract_streamlines: Original (x,y,z) streamlines grouped by tract_id.
        gt_labels: Ground-truth labels from HDF5 (one per streamline).
        pred_labels: Predicted labels from model inference.
        affine: NIfTI affine matrix for voxelization.

    Returns:
        Dictionary with per-bundle wDice and mean wDice.
    """
    # Flatten all original streamlines in the same order as HDF5
    # (sorted by tract_id, sequential within each tract)
    all_streamlines = []
    for tract_id in sorted(tract_streamlines.keys()):
        all_streamlines.extend(tract_streamlines[tract_id])

    assert len(all_streamlines) == len(gt_labels), (
        f"Streamline count mismatch: {len(all_streamlines)} originals vs "
        f"{len(gt_labels)} in HDF5"
    )

    # Group original streamlines by PREDICTED label
    pred_groups: Dict[int, List[np.ndarray]] = defaultdict(list)
    for sl, pred_label in zip(all_streamlines, pred_labels):
        pred_groups[pred_label].append(sl)

    # Group original streamlines by REFERENCE label
    ref_groups: Dict[int, List[np.ndarray]] = defaultdict(list)
    for sl, gt_label in zip(all_streamlines, gt_labels):
        ref_groups[gt_label].append(sl)

    # Compute per-bundle wDice and Dice
    results = {}
    per_bundle_wdice = []
    per_bundle_dice = []

    all_tract_ids = sorted(set(list(ref_groups.keys()) + list(pred_groups.keys())))

    for tract_id in all_tract_ids:
        tract_name = ID_TO_TRACT.get(tract_id, f"tract_{tract_id}")

        ref_sls = ref_groups.get(tract_id, [])
        pred_sls = pred_groups.get(tract_id, [])

        if not ref_sls and not pred_sls:
            continue

        density_ref = voxelize_streamlines(ref_sls, affine) if ref_sls else {}
        density_pred = voxelize_streamlines(pred_sls, affine) if pred_sls else {}
        
        wdice = compute_wdice(density_ref, density_pred)
        dice = compute_dice(density_ref, density_pred)

        results[tract_name] = wdice
        results[f"{tract_name}_Dice"] = dice
        per_bundle_wdice.append(wdice)
        per_bundle_dice.append(dice)

    results['mean_wDice'] = float(np.mean(per_bundle_wdice)) if per_bundle_wdice else 0.0
    results['mean_Dice'] = float(np.mean(per_bundle_dice)) if per_bundle_dice else 0.0
    return results


# ──────────────────────────── Plotting ────────────────────────────────────── #

def plot_wdice_results(
    results: Dict[str, float],
    save_path: str,
    title: str = "Per-Bundle wDice Scores"
):
    """Generate a horizontal bar chart of per-bundle wDice scores."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Exclude 'mean_wDice' from per-bundle plot
    bundles = {k: v for k, v in results.items() if k != 'mean_wDice'}
    if not bundles:
        return

    # Sort by wDice value
    sorted_bundles = sorted(bundles.items(), key=lambda x: x[1], reverse=True)
    names = [b[0] for b in sorted_bundles]
    scores = [b[1] for b in sorted_bundles]

    fig, ax = plt.subplots(figsize=(10, max(8, len(names) * 0.35)))

    colors = plt.cm.RdYlGn(np.array(scores))
    bars = ax.barh(range(len(names)), scores, color=colors, edgecolor='gray', linewidth=0.5)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('wDice Score', fontsize=12)
    ax.set_title(f"{title}\n(Mean wDice = {results.get('mean_wDice', 0):.4f})", fontsize=13)
    ax.set_xlim(0, 1.05)
    ax.axvline(x=results.get('mean_wDice', 0), color='navy', linestyle='--',
               linewidth=1.5, label=f"Mean = {results.get('mean_wDice', 0):.4f}")
    ax.legend(loc='lower right')
    ax.invert_yaxis()

    # Add score labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.01, i, f"{score:.4f}", va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved wDice plot to {save_path}")


# ─────────────────────────────── Main ─────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description='Compute volumetric weighted Dice (wDice) for bundle classification'
    )
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, default='sequences/testset',
                        help='Directory with test HDF5 files')
    parser.add_argument('--tractoinferno_dir', type=str,
                        default='/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives',
                        help='Path to Tractoinferno derivatives directory')
    parser.add_argument('--scope', type=str, default='testset',
                        choices=['trainset', 'validset', 'testset'],
                        help='Dataset scope to evaluate')
    parser.add_argument('--experiment_name', type=str, default='best_model',
                        help='Experiment name')
    parser.add_argument('--output_dir', type=str, default='tests/wdice_results',
                        help='Directory to save wDice results')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size for inference')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision')
    parser.add_argument('--preds_dir', type=str, default=None,
                        help='Directory to save/load predictions to avoid recomputing')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"\n{'='*60}")
    print(f"WEIGHTED DICE (wDice) EVALUATION")
    print(f"{'='*60}")
    print(f"\nLoading model from {args.checkpoint}")
    cfg = DEFAULT_CONFIG
    model, checkpoint = load_model(args.checkpoint, cfg, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Get test HDF5 files
    test_dir = pathlib.Path(args.test_dir)
    hdf5_files = sorted(test_dir.glob('*.hdf5'))
    print(f"\nFound {len(hdf5_files)} test HDF5 files in {test_dir}")

    if not hdf5_files:
        print("ERROR: No HDF5 files found!")
        return

    # Set up dataset handler for loading original .trk files
    dataset_handler = Tractoinferno_handler(args.tractoinferno_dir, scope=args.scope)
    subjects_data = {s['subject']: s for s in dataset_handler.get_data()}

    # Process each subject
    all_subject_results = {}
    aggregate_wdice = defaultdict(list)

    for i, hdf5_path in enumerate(hdf5_files):
        subject_name = hdf5_path.stem  # e.g., "sub-1006"
        print(f"\n{'─'*60}")
        print(f"[{i+1}/{len(hdf5_files)}] Processing {subject_name}")
        print(f"{'─'*60}")

        # Find matching subject in Tractoinferno
        if subject_name not in subjects_data:
            print(f"  ⚠ Subject {subject_name} not found in Tractoinferno {args.scope}, skipping")
            continue

        subject_info = subjects_data[subject_name]
        subject_path = pathlib.Path(subject_info['T1w']).parent.parent
        mri_path = subject_info['T1w']

        # Step 1: Run model inference on HDF5
        gt_labels, pred_labels, _ = run_inference_per_subject(
            model, str(hdf5_path), device,
            batch_size=args.batch_size,
            use_amp=not args.no_amp,
            preds_dir=args.preds_dir
        )
        accuracy = np.mean(np.array(gt_labels) == np.array(pred_labels)) * 100
        print(f"  📈 Subject accuracy: {accuracy:.2f}%")

        # Step 2: Load original .trk streamlines
        print(f"  📂 Loading original .trk streamlines...")
        tract_streamlines, affine = load_subject_streamlines(subject_path, mri_path)
        total_original = sum(len(sls) for sls in tract_streamlines.values())
        print(f"  📊 Loaded {total_original} original streamlines from {len(tract_streamlines)} bundles")

        # Step 3: Compute wDice
        print(f"  🎲 Voxelizing and computing wDice...")
        subject_results = compute_subject_wdice(
            tract_streamlines, gt_labels, pred_labels, affine
        )

        all_subject_results[subject_name] = subject_results
        for bundle, score in subject_results.items():
            aggregate_wdice[bundle].append(score)

        print(f"  ✅ Mean wDice: {subject_results['mean_wDice']:.4f}  |  Mean Dice: {subject_results['mean_Dice']:.4f}")

    # ── Aggregate results across subjects ──
    if not all_subject_results:
        print("\nNo subjects processed!")
        return

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")

    # Compute mean wDice and mean Dice per bundle across all subjects
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

    # Save per-subject results as JSON
    json_path = os.path.join(args.output_dir, 'wdice_per_subject.json')
    with open(json_path, 'w') as f:
        json.dump(all_subject_results, f, indent=2)
    print(f"\n💾 Per-subject results saved to {json_path}")

    # Save aggregate results as JSON
    agg_json_path = os.path.join(args.output_dir, 'wdice_aggregate.json')
    with open(agg_json_path, 'w') as f:
        json.dump(mean_results, f, indent=2)
    print(f"💾 Aggregate results saved to {agg_json_path}")

    # Save CSV summary
    try:
        import pandas as pd
        rows = []
        for subj, results in all_subject_results.items():
            row = {'subject': subj}
            row.update(results)
            rows.append(row)
        df = pd.DataFrame(rows)
        csv_path = os.path.join(args.output_dir, 'wdice_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"💾 CSV summary saved to {csv_path}")
    except ImportError:
        print("  (pandas not available, skipping CSV export)")

    # Generate plots
    wdice_plot_data = {k: v for k, v in mean_results.items() if not k.endswith('_Dice')}
    plot_wdice_results(
        wdice_plot_data,
        os.path.join(args.output_dir, 'wdice_per_bundle.png'),
        title="Per-Bundle wDice (Averaged Across Subjects)"
    )
    
    dice_plot_data = {k.replace('_Dice', ''): v for k, v in mean_results.items() if k.endswith('_Dice') or k == 'mean_Dice'}
    if 'mean_Dice' in mean_results:
        dice_plot_data['mean_wDice'] = mean_results['mean_Dice'] # Rename for plot function compat
    
    plot_wdice_results(
        dice_plot_data,
        os.path.join(args.output_dir, 'dice_per_bundle.png'),
        title="Per-Bundle Dice (Averaged Across Subjects)"
    )

    # Per-subject wDice and Dice bar charts
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        subjects = list(all_subject_results.keys())
        mean_wdice_scores = [all_subject_results[s]['mean_wDice'] for s in subjects]
        mean_dice_scores = [all_subject_results[s]['mean_Dice'] for s in subjects]

        fig, axes = plt.subplots(1, 2, figsize=(max(14, len(subjects)), 5))
        
        # wDice plot
        colors = plt.cm.RdYlGn(np.array(mean_wdice_scores))
        axes[0].bar(range(len(subjects)), mean_wdice_scores, color=colors, edgecolor='gray')
        axes[0].set_xticks(range(len(subjects)))
        axes[0].set_xticklabels(subjects, rotation=45, ha='right', fontsize=8)
        axes[0].set_ylabel('Mean wDice', fontsize=12)
        axes[0].set_title('Mean wDice per Subject', fontsize=13)
        axes[0].set_ylim(0, 1.05)
        axes[0].axhline(
            y=mean_results['mean_wDice'], color='navy', linestyle='--',
            linewidth=1.5, label=f"Overall Mean = {mean_results['mean_wDice']:.4f}"
        )
        axes[0].legend()
        
        # Dice plot
        colors = plt.cm.RdYlGn(np.array(mean_dice_scores))
        axes[1].bar(range(len(subjects)), mean_dice_scores, color=colors, edgecolor='gray')
        axes[1].set_xticks(range(len(subjects)))
        axes[1].set_xticklabels(subjects, rotation=45, ha='right', fontsize=8)
        axes[1].set_ylabel('Mean Dice', fontsize=12)
        axes[1].set_title('Mean Dice per Subject', fontsize=13)
        axes[1].set_ylim(0, 1.05)
        axes[1].axhline(
            y=mean_results.get('mean_Dice', 0.0), color='navy', linestyle='--',
            linewidth=1.5, label=f"Overall Mean = {mean_results.get('mean_Dice', 0.0):.4f}"
        )
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, 'metrics_per_subject.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print(f"  📊 Saved per-subject metrics plot")
    except Exception as e:
        print(f"  ⚠ Could not generate per-subject plot: {e}")

    print(f"\n✅ wDice evaluation complete!")


if __name__ == '__main__':
    main()
