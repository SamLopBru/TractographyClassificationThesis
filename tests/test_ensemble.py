"""
Ensemble test script — averages softmax probabilities from multiple models.

Loads N trained checkpoints, runs inference with each, averages the per-sample
softmax probabilities, and evaluates the ensemble using the same metrics and
plots as test.py.

Usage:
    uv run tests/test_ensemble.py \
        --checkpoints \
            checkpoints/model_a/best_model.pt \
            checkpoints/model_b/best_model.pt \
            checkpoints/model_c/best_model.pt \
        --output_dir tests/test_results/ensemble
"""

import torch
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test import (
    load_model,
    evaluate_model,
    compute_metrics,
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_class_distribution,
    plot_prediction_confidence,
    plot_misclassification_analysis,
)
from utils.dataloader import StreamlineDataset, streamline_collate_fn
from src.config import DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble evaluation — average softmax probabilities from multiple models"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Paths to model checkpoints (best_model.pt files)",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="sequences/testset",
        help="Directory containing test HDF5 files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tests/test_results/ensemble",
        help="Directory to save ensemble results",
    )
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--sampling_pct",
        type=float,
        default=0.25,
        help="Percentage of test data to use (1.0 = all)",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp

    print(f"Using device: {device}")
    print(f"Ensemble of {len(args.checkpoints)} models:")
    for i, cp in enumerate(args.checkpoints):
        print(f"  [{i+1}] {cp}")

    # ── Load test data (once) ────────────────────────────────────────
    test_dir = Path(args.test_dir)
    test_files = sorted([str(f) for f in test_dir.glob("*.hdf5")])
    print(f"\nFound {len(test_files)} test HDF5 files in {test_dir}")
    if not test_files:
        print("ERROR: No HDF5 files found!")
        return

    test_dataset = StreamlineDataset(
        test_files,
        sampling_percentage=args.sampling_pct,
        full_sample_threshold=100_000,
    )
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=streamline_collate_fn,
    )

    # ── Run inference with each model ────────────────────────────────
    all_model_probs = []
    individual_accs = []
    labels = None

    for i, ckpt_path in enumerate(args.checkpoints):
        print(f"\n{'─'*60}")
        print(f"Model [{i+1}/{len(args.checkpoints)}]: {ckpt_path}")
        print("─" * 60)

        cfg = DEFAULT_CONFIG.__class__()  # fresh config for each model
        model, _ = load_model(ckpt_path, cfg, device)

        model_labels, model_preds, model_probs = evaluate_model(
            model, test_loader, device, use_amp=use_amp
        )

        # Verify labels are consistent across models
        if labels is None:
            labels = model_labels
        else:
            assert np.array_equal(labels, model_labels), \
                "Label mismatch between models — is the test set deterministic?"

        acc = (model_labels == model_preds).mean() * 100
        individual_accs.append(acc)
        print(f"  Individual accuracy: {acc:.2f}%")

        all_model_probs.append(model_probs)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # ── Average probabilities ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ENSEMBLE RESULTS")
    print("=" * 60)

    # Stack and average: (N_models, N_samples, N_classes) → (N_samples, N_classes)
    ensemble_probs = np.mean(np.stack(all_model_probs, axis=0), axis=0)
    ensemble_preds = ensemble_probs.argmax(axis=1)

    num_classes = ensemble_probs.shape[1]
    class_names = [f"Bundle_{i}" for i in range(num_classes)]

    # ── Compute and display metrics ──────────────────────────────────
    metrics = compute_metrics(labels, ensemble_preds, ensemble_probs, num_classes, class_names)

    print(f"\n📊 Individual model accuracies:")
    for i, (acc, cp) in enumerate(zip(individual_accs, args.checkpoints)):
        name = os.path.basename(os.path.dirname(cp))
        print(f"   [{i+1}] {name}: {acc:.2f}%")

    print(f"\n🏆 Ensemble Metrics ({len(args.checkpoints)} models):")
    print(f"   Accuracy:            {metrics['accuracy']:.2f}%")
    print(f"   Precision (macro):   {metrics['precision_macro']:.2f}%")
    print(f"   Recall (macro):      {metrics['recall_macro']:.2f}%")
    print(f"   F1 Score (macro):    {metrics['f1_macro']:.2f}%")
    print(f"   F1 Score (weighted): {metrics['f1_weighted']:.2f}%")
    if "top_3_accuracy" in metrics:
        print(f"   Top-3 Accuracy:      {metrics['top_3_accuracy']:.2f}%")
    if "top_5_accuracy" in metrics:
        print(f"   Top-5 Accuracy:      {metrics['top_5_accuracy']:.2f}%")

    print(f"\n📋 Classification Report:\n")
    print(metrics["classification_report"])

    # ── Save results ─────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # Classification report
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(metrics["classification_report"])
    print(f"💾 Classification report saved to {report_path}")

    # Plots
    print("\n📈 Generating visualization plots...")

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        os.path.join(args.output_dir, "confusion_matrix.png"),
        class_names=class_names,
    )
    plot_per_class_metrics(
        metrics,
        os.path.join(args.output_dir, "per_class_metrics.png"),
        class_names=class_names,
    )
    plot_class_distribution(
        metrics["samples_per_class"],
        os.path.join(args.output_dir, "class_distribution.png"),
        class_names=class_names,
    )
    plot_prediction_confidence(
        labels,
        ensemble_probs,
        ensemble_preds,
        os.path.join(args.output_dir, "confidence_analysis.png"),
    )
    plot_misclassification_analysis(
        labels,
        ensemble_preds,
        os.path.join(args.output_dir, "misclassification_analysis.png"),
        class_names=class_names,
    )

    # Metrics JSON
    import json

    metrics_to_save = {
        "ensemble_size": len(args.checkpoints),
        "checkpoints": args.checkpoints,
        "individual_accuracies": {
            os.path.basename(os.path.dirname(cp)): acc
            for cp, acc in zip(args.checkpoints, individual_accs)
        },
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_macro": metrics["recall_macro"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "total_samples": len(labels),
        "total_correct": int((labels == ensemble_preds).sum()),
    }
    if "top_3_accuracy" in metrics:
        metrics_to_save["top_3_accuracy"] = metrics["top_3_accuracy"]
    if "top_5_accuracy" in metrics:
        metrics_to_save["top_5_accuracy"] = metrics["top_5_accuracy"]

    metrics_file = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nSaved metrics to {metrics_file}")

    print(f"\n{'='*60}")
    print("ENSEMBLE EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\n📁 All results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
