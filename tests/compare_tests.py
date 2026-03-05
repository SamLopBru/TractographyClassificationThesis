"""
Compare test results across multiple experiments.

Reads all test_metrics.json and classification_report.txt files from
test_results/ subdirectories and generates:
  - A summary comparison table (printed + CSV)
  - Bar charts comparing overall metrics across experiments
  - A per-class F1 heatmap across experiments
  - A per-class delta chart (best vs worst F1 per class)

Usage:
    python compare_tests.py
    python compare_tests.py --results_dir tests/test_results
    python compare_tests.py --output_dir tests/test_results/comparison
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def load_metrics(results_dir: Path) -> Dict[str, dict]:
    """Load test_metrics.json from every subdirectory of *results_dir*."""
    metrics = {}
    for subdir in sorted(results_dir.iterdir()):
        json_path = subdir / "test_metrics.json"
        if subdir.is_dir() and json_path.exists():
            with open(json_path) as f:
                metrics[subdir.name] = json.load(f)
    return metrics


def parse_classification_report(report_path: Path) -> pd.DataFrame:
    """
    Parse a sklearn-style classification_report.txt into a DataFrame
    with columns: class, precision, recall, f1, support.
    """
    rows = []
    with open(report_path) as f:
        for line in f:
            line = line.rstrip()
            # Match lines like "    Bundle_0       1.00      1.00      1.00    736803"
            m = re.match(
                r"\s+(Bundle_\d+|macro avg|weighted avg|accuracy)\s+"
                r"(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)",
                line,
            )
            if m:
                rows.append({
                    "class": m.group(1),
                    "precision": float(m.group(2)),
                    "recall": float(m.group(3)),
                    "f1": float(m.group(4)),
                    "support": int(m.group(5)),
                })
            # Handle the accuracy line which has a different format
            m_acc = re.match(
                r"\s+accuracy\s+(\d+\.\d+)\s+(\d+)",
                line,
            )
            if m_acc:
                rows.append({
                    "class": "accuracy",
                    "precision": None,
                    "recall": None,
                    "f1": float(m_acc.group(1)),
                    "support": int(m_acc.group(2)),
                })
    return pd.DataFrame(rows)


def load_all_reports(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load classification reports from every subdirectory."""
    reports = {}
    for subdir in sorted(results_dir.iterdir()):
        report_path = subdir / "classification_report.txt"
        if subdir.is_dir() and report_path.exists():
            reports[subdir.name] = parse_classification_report(report_path)
    return reports


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

OVERVIEW_COLS = [
    "accuracy", "f1_macro", "f1_weighted",
    "precision_macro", "recall_macro",
    "top_3_accuracy", "top_5_accuracy",
]


def build_summary_table(metrics: Dict[str, dict]) -> pd.DataFrame:
    """Build a DataFrame where rows=experiments, cols=metrics."""
    rows = []
    for name, m in metrics.items():
        row = {"experiment": name}
        for col in OVERVIEW_COLS:
            row[col] = m.get(col)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("experiment")
    return df


def highlight_best(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the best value in each column marked with ✅."""
    df_str = df.copy().astype(str)
    for col in df.columns:
        best_idx = df[col].idxmax()
        df_str.at[best_idx, col] = f"{df.at[best_idx, col]:.2f} ✅"
    return df_str

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def short_name(name: str) -> str:
    """Shorten long experiment names for plot labels."""
    name = name.replace("transformer_", "")
    # Abbreviate further if too long
    if len(name) > 25:
        parts = name.split("_")
        name = "_".join(parts[:4])
    return name


def plot_overview_bars(df: pd.DataFrame, output_dir: Path):
    """Bar chart comparing the main metrics across experiments."""
    metrics_to_plot = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    labels = [short_name(n) for n in df.index]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4 * len(metrics_to_plot), 5))
    colors = sns.color_palette("viridis", n_colors=len(df))

    for ax, metric in zip(axes, metrics_to_plot):
        values = df[metric].values
        bars = ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(metric.replace("_", " ").title(), fontweight="bold")

        # Narrow y-range to highlight differences
        vmin, vmax = values.min(), values.max()
        margin = max((vmax - vmin) * 0.6, 0.3)
        ax.set_ylim(vmin - margin, vmax + margin)

        # Annotate values
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + margin * 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7,
            )

    fig.suptitle("Overall Metric Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "overview_comparison.png")
    plt.close(fig)
    print(f"  ✓ Saved overview_comparison.png")


def plot_per_class_f1_heatmap(reports: Dict[str, pd.DataFrame], output_dir: Path):
    """Heatmap of per-class F1 scores across experiments."""
    # Build matrix: rows=classes, cols=experiments
    class_names = []
    exp_names = list(reports.keys())
    data = {}

    for exp_name, df in reports.items():
        bundle_rows = df[df["class"].str.startswith("Bundle_")]
        for _, row in bundle_rows.iterrows():
            cname = row["class"]
            if cname not in data:
                data[cname] = {}
                class_names.append(cname)
            data[cname][exp_name] = row["f1"]

    matrix = pd.DataFrame(data).T
    matrix = matrix[exp_names]  # keep experiment order
    matrix.columns = [short_name(n) for n in matrix.columns]

    fig, ax = plt.subplots(figsize=(max(8, len(exp_names) * 2.5), max(8, len(class_names) * 0.35)))
    sns.heatmap(
        matrix.astype(float), annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=0.4, vmax=1.0,
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"label": "F1-Score"},
    )
    ax.set_title("Per-Class F1-Score Across Experiments", fontsize=14, fontweight="bold")
    ax.set_ylabel("Bundle Class")
    ax.set_xlabel("Experiment")
    plt.tight_layout()
    fig.savefig(output_dir / "per_class_f1_heatmap.png")
    plt.close(fig)
    print(f"  ✓ Saved per_class_f1_heatmap.png")


def plot_f1_delta(reports: Dict[str, pd.DataFrame], output_dir: Path):
    """Show F1 range (best - worst) per class to highlight where models disagree most."""
    exp_names = list(reports.keys())
    class_f1: Dict[str, List[float]] = {}

    for exp_name, df in reports.items():
        bundle_rows = df[df["class"].str.startswith("Bundle_")]
        for _, row in bundle_rows.iterrows():
            cname = row["class"]
            class_f1.setdefault(cname, []).append(row["f1"])

    classes = list(class_f1.keys())
    deltas = [max(class_f1[c]) - min(class_f1[c]) for c in classes]

    # Sort by delta descending
    order = np.argsort(deltas)[::-1]
    classes = [classes[i] for i in order]
    deltas = [deltas[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(5, len(classes) * 0.3)))
    colors = ["#e74c3c" if d > 0.03 else "#2ecc71" if d < 0.01 else "#f39c12" for d in deltas]
    ax.barh(range(len(classes)), deltas, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("F1 Range (max - min across experiments)")
    ax.set_title("Per-Class F1 Variability Across Experiments", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    # Add value labels
    for i, d in enumerate(deltas):
        ax.text(d + 0.002, i, f"{d:.3f}", va="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="> 0.03 (high variability)"),
        Patch(facecolor="#f39c12", label="0.01–0.03 (moderate)"),
        Patch(facecolor="#2ecc71", label="< 0.01 (stable)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "per_class_f1_delta.png")
    plt.close(fig)
    print(f"  ✓ Saved per_class_f1_delta.png")


def plot_radar(df: pd.DataFrame, output_dir: Path):
    """Radar / spider chart to compare experiment profiles."""
    metrics = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    labels = [m.replace("_", " ").title() for m in metrics]
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = sns.color_palette("husl", n_colors=len(df))

    for (exp_name, row), color in zip(df.iterrows(), colors):
        values = [row[m] for m in metrics]
        values += values[:1]  # close the polygon
        ax.plot(angles, values, "o-", linewidth=2, label=short_name(exp_name), color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    # Narrow y-range
    all_vals = df[metrics].values.flatten()
    ymin = max(all_vals.min() - 1, 0)
    ymax = min(all_vals.max() + 1, 100)
    ax.set_ylim(ymin, ymax)

    ax.set_title("Experiment Profile Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "radar_comparison.png")
    plt.close(fig)
    print(f"  ✓ Saved radar_comparison.png")


def plot_best_model_per_class(reports: Dict[str, pd.DataFrame], output_dir: Path):
    """For each class, show which model achieves the best F1."""
    exp_names = list(reports.keys())
    class_best: Dict[str, Tuple[str, float]] = {}

    for exp_name, df in reports.items():
        bundle_rows = df[df["class"].str.startswith("Bundle_")]
        for _, row in bundle_rows.iterrows():
            cname = row["class"]
            f1 = row["f1"]
            if cname not in class_best or f1 > class_best[cname][1]:
                class_best[cname] = (exp_name, f1)

    # Count how many classes each model wins
    win_counts: Dict[str, int] = {n: 0 for n in exp_names}
    for cname, (winner, _) in class_best.items():
        win_counts[winner] += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [short_name(n) for n in exp_names]
    counts = [win_counts[n] for n in exp_names]
    colors = sns.color_palette("viridis", n_colors=len(exp_names))

    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=0.5)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(c), ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_ylabel("Number of Classes (Best F1)")
    ax.set_title("Best Model Per Class (by F1-Score)", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "best_model_per_class.png")
    plt.close(fig)
    print(f"  ✓ Saved best_model_per_class.png")


def plot_per_class_f1_scatter(reports: Dict[str, pd.DataFrame], output_dir: Path):
    """Scatter / dot plot of per-class F1 with class names on y-axis.

    Each experiment is shown with a distinct marker and color so the
    plot resembles the classic per-bundle comparison dot chart.
    """
    markers = ["x", "o", "s", "D", "^", "v", "P", "*"]
    palette = sns.color_palette("bright", n_colors=len(reports))

    # Collect per-class F1 for each experiment
    exp_names = list(reports.keys())
    class_names: List[str] = []
    class_f1: Dict[str, Dict[str, float]] = {}  # class -> {exp: f1}

    for exp_name, df in reports.items():
        bundle_rows = df[df["class"].str.startswith("Bundle_")]
        for _, row in bundle_rows.iterrows():
            cname = row["class"]
            if cname not in class_f1:
                class_f1[cname] = {}
                class_names.append(cname)
            class_f1[cname][exp_name] = row["f1"]

    # Sort classes by name (numerical order)
    class_names = sorted(class_names, key=lambda c: int(c.split("_")[1]))
    y_positions = {c: i for i, c in enumerate(class_names)}

    fig, ax = plt.subplots(figsize=(10, max(6, len(class_names) * 0.35)))

    for idx, exp_name in enumerate(exp_names):
        xs, ys = [], []
        for cname in class_names:
            if exp_name in class_f1.get(cname, {}):
                xs.append(class_f1[cname][exp_name])
                ys.append(y_positions[cname])
        ax.scatter(
            xs, ys,
            marker=markers[idx % len(markers)],
            color=palette[idx % len(palette)],
            s=70,
            label=short_name(exp_name),
            edgecolors="white" if markers[idx % len(markers)] != "x" else None,
            linewidths=0.5,
            zorder=3,
        )

    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.invert_yaxis()  # Bundle_0 at top
    ax.set_xlabel("F1-Score", fontsize=12)
    ax.set_title("Per-Class F1-Score Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_dir / "per_class_f1_scatter.png")
    plt.close(fig)
    print(f"  ✓ Saved per_class_f1_scatter.png")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare test results across experiments")
    parser.add_argument(
        "--results_dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "test_results"),
        help="Directory containing experiment subdirectories (default: tests/test_results)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save comparison outputs (default: <results_dir>/comparison)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Results directory: {results_dir}")
    print(f"📁 Output directory:  {output_dir}")

    # ---- Load data ----
    metrics = load_metrics(results_dir)
    reports = load_all_reports(results_dir)

    if not metrics:
        print("❌ No test_metrics.json files found. Exiting.")
        sys.exit(1)

    print(f"\n🔍 Found {len(metrics)} experiment(s): {', '.join(metrics.keys())}\n")

    # ---- Summary table ----
    summary_df = build_summary_table(metrics)
    print("=" * 80)
    print("OVERALL METRICS COMPARISON")
    print("=" * 80)
    highlighted = highlight_best(summary_df)
    print(highlighted.to_string())
    print()

    # Save CSV
    summary_df.to_csv(output_dir / "summary_comparison.csv")
    print(f"  ✓ Saved summary_comparison.csv")

    # ---- Per-class best-model table ----
    if reports:
        exp_names = list(reports.keys())
        class_data = {}
        for exp_name, df in reports.items():
            bundle_rows = df[df["class"].str.startswith("Bundle_")]
            for _, row in bundle_rows.iterrows():
                cname = row["class"]
                if cname not in class_data:
                    class_data[cname] = {}
                class_data[cname][exp_name] = row["f1"]

        per_class_df = pd.DataFrame(class_data).T
        per_class_df.columns = [short_name(n) for n in per_class_df.columns]
        per_class_df["best_model"] = per_class_df.idxmax(axis=1)
        per_class_df["best_f1"] = per_class_df.drop(columns=["best_model"]).max(axis=1)
        per_class_df["worst_f1"] = per_class_df.drop(columns=["best_model", "best_f1"]).min(axis=1)
        per_class_df["delta"] = per_class_df["best_f1"] - per_class_df["worst_f1"]

        print("=" * 80)
        print("PER-CLASS COMPARISON (F1-Score)")
        print("=" * 80)
        print(per_class_df.to_string())
        print()

        per_class_df.to_csv(output_dir / "per_class_comparison.csv")
        print(f"  ✓ Saved per_class_comparison.csv")

    # ---- Plots ----
    print("\n📊 Generating plots...")
    plot_overview_bars(summary_df, output_dir)
    if reports:
        plot_per_class_f1_heatmap(reports, output_dir)
        plot_f1_delta(reports, output_dir)
        plot_best_model_per_class(reports, output_dir)
        plot_per_class_f1_scatter(reports, output_dir)
    plot_radar(summary_df, output_dir)

    print(f"\n✅ All outputs saved to: {output_dir}")
    print(f"   Files: {', '.join(f.name for f in sorted(output_dir.iterdir()))}")


if __name__ == "__main__":
    main()
