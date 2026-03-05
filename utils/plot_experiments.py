"""
Visualize hyperparameter configurations and results from experiments.csv.

Creates a multi-panel figure showing:
  1. F1 Score by experiment (horizontal bar chart)
  2. F1 vs number of parameters (scatter)
  3. Hyperparameter heatmap (normalized values)
  4. F1 by key hyperparameter groups (box/strip plots)

Usage:
    python utils/plot_experiments.py
    python utils/plot_experiments.py --csv path/to/experiments.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import argparse
import os

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'text.color': '#c9d1d9',
    'grid.color': '#21262d',
    'grid.alpha': 0.6,
    'font.family': 'sans-serif',
    'font.size': 10,
})

ACCENT = '#58a6ff'
GOLD   = '#f0c040'
GREEN  = '#3fb950'
RED    = '#f85149'
PURPLE = '#bc8cff'
ORANGE = '#f0883e'


def short_label(row):
    """Build a concise, unique label for each experiment row."""
    parts = [row['encoder_type'][:3].upper()]
    parts.append(f"d{int(row['d_model'])}")
    parts.append(f"L{int(row['num_layers'])}")
    parts.append(f"h{int(row['nhead'])}")
    parts.append(row['pooling'])
    if row['dropout'] != 0.1:
        parts.append(f"drop{row['dropout']}")
    warmup = row.get('warmup_steps', '')
    if pd.notna(warmup) and warmup != '' and float(warmup) > 0:
        parts.append(f"w{int(float(warmup))}")
    notes = row.get('NOTAS', '')
    if pd.notna(notes) and str(notes).strip():
        parts.append(str(notes).strip())
    return ' | '.join(parts)


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Drop completely empty rows
    df = df.dropna(subset=['best_val_f1'])
    df['best_val_f1'] = pd.to_numeric(df['best_val_f1'], errors='coerce')
    df['best_val_acc'] = pd.to_numeric(df['best_val_acc'], errors='coerce')
    df['parameters'] = pd.to_numeric(df['parameters'], errors='coerce')
    df['d_model'] = pd.to_numeric(df['d_model'], errors='coerce')
    df['num_layers'] = pd.to_numeric(df['num_layers'], errors='coerce')
    df['nhead'] = pd.to_numeric(df['nhead'], errors='coerce')
    df['dropout'] = pd.to_numeric(df['dropout'], errors='coerce')
    df['warmup_steps'] = pd.to_numeric(df['warmup_steps'], errors='coerce')
    df['dim_feedforward'] = pd.to_numeric(df['dim_feedforward'], errors='coerce')
    df['batch_size'] = pd.to_numeric(df['batch_size'], errors='coerce')
    df['lr'] = pd.to_numeric(df['lr'], errors='coerce')
    df['epoch_sampling_pct'] = pd.to_numeric(df['epoch_sampling_pct'], errors='coerce')
    df['label'] = df.apply(short_label, axis=1)
    df = df.sort_values('best_val_f1', ascending=True).reset_index(drop=True)
    return df


def plot_f1_bars(ax, df):
    """Panel 1: Horizontal bar chart of F1 scores per experiment."""
    colors = []
    best_f1 = df['best_val_f1'].max()
    for _, row in df.iterrows():
        if row['encoder_type'] == 'lstm':
            colors.append(ORANGE)
        elif row['best_val_f1'] == best_f1:
            colors.append(GOLD)
        else:
            colors.append(ACCENT)

    bars = ax.barh(range(len(df)), df['best_val_f1'], color=colors, 
                   edgecolor='none', height=0.7, alpha=0.9)
    
    # Value labels
    for i, (val, bar) in enumerate(zip(df['best_val_f1'], bars)):
        ax.text(val + 0.02, i, f'{val:.2f}', va='center', fontsize=7.5,
                color=GOLD if val == best_f1 else '#8b949e', fontweight='bold' if val == best_f1 else 'normal')

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['label'], fontsize=7)
    ax.set_xlabel('Best Validation F1 (%)')
    ax.set_title('F1 Score by Configuration', fontweight='bold', fontsize=12)
    ax.set_xlim(df['best_val_f1'].min() - 0.3, df['best_val_f1'].max() + 0.5)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ACCENT, label='Transformer'),
        Patch(facecolor=ORANGE, label='LSTM'),
        Patch(facecolor=GOLD, label='Best'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
              facecolor='#161b22', edgecolor='#30363d')


def plot_f1_vs_params(ax, df):
    """Panel 2: Scatter plot of F1 vs number of parameters."""
    for enc_type, color, marker in [('transformer', ACCENT, 'o'), ('lstm', ORANGE, 's')]:
        mask = df['encoder_type'] == enc_type
        sub = df[mask]
        ax.scatter(sub['parameters'] / 1e6, sub['best_val_f1'], 
                   c=color, marker=marker, s=60, alpha=0.85, edgecolors='white',
                   linewidths=0.5, label=enc_type.upper(), zorder=3)
    
    # Highlight best
    best_idx = df['best_val_f1'].idxmax()
    best = df.loc[best_idx]
    ax.scatter(best['parameters'] / 1e6, best['best_val_f1'], 
               c=GOLD, marker='*', s=200, edgecolors='white', linewidths=0.8, 
               zorder=4, label='Best')
    ax.annotate(f"F1={best['best_val_f1']:.2f}", 
                xy=(best['parameters'] / 1e6, best['best_val_f1']),
                xytext=(10, 10), textcoords='offset points', fontsize=8,
                color=GOLD, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=GOLD, lw=1))

    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('Best Validation F1 (%)')
    ax.set_title('F1 vs Model Size', fontweight='bold', fontsize=12)
    ax.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, linestyle='--', alpha=0.3)


def plot_heatmap(ax, df):
    """Panel 3: Heatmap of key hyperparameters (normalized) sorted by F1."""
    hp_cols = ['d_model', 'num_layers', 'nhead', 'dim_feedforward', 'dropout', 
               'lr', 'batch_size', 'warmup_steps', 'epoch_sampling_pct']
    
    hp_data = df[hp_cols].copy()
    hp_data = hp_data.fillna(0)
    
    # Normalize each column to [0, 1]
    for col in hp_cols:
        col_min = hp_data[col].min()
        col_max = hp_data[col].max()
        if col_max > col_min:
            hp_data[col] = (hp_data[col] - col_min) / (col_max - col_min)
        else:
            hp_data[col] = 0.5
    
    im = ax.imshow(hp_data.values.T, aspect='auto', cmap='cool', 
                   interpolation='nearest', alpha=0.85)
    
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{f1:.1f}" for f1 in df['best_val_f1']], 
                       rotation=45, ha='right', fontsize=6.5)
    ax.set_yticks(range(len(hp_cols)))
    col_labels = ['d_model', 'layers', 'heads', 'dim_ff', 'dropout', 
                  'lr', 'batch', 'warmup', 'epoch_samp']
    ax.set_yticklabels(col_labels, fontsize=8)
    ax.set_xlabel('Experiments (sorted by F1 ↑)')
    ax.set_title('Hyperparameter Heatmap', fontweight='bold', fontsize=12)
    
    # Add actual values as text
    for i in range(len(hp_cols)):
        for j in range(len(df)):
            val = df.iloc[j][hp_cols[i]]
            if hp_cols[i] == 'lr':
                text = f'{val:.0e}' if pd.notna(val) else ''
            elif hp_cols[i] in ['dropout', 'epoch_sampling_pct']:
                text = f'{val:.2f}' if pd.notna(val) else ''
            else:
                text = f'{int(val)}' if pd.notna(val) and val > 0 else '0'
            ax.text(j, i, text, ha='center', va='center', fontsize=5.5,
                    color='black' if hp_data.values.T[i, j] > 0.5 else 'white',
                    fontweight='bold')
    
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04, label='Normalized')


def plot_grouped_f1(ax, df):
    """Panel 4: F1 by key hyperparameter choices."""
    groups = {
        'Pooling': ('pooling', ['mean', 'cls', 'max']),
        'Encoder': ('encoder_type', ['transformer', 'lstm']),
        'd_model': ('d_model', [256, 384, 512]),
        'Layers': ('num_layers', [2, 4, 6, 10]),
    }
    
    positions = []
    labels = []
    pos = 0
    colors_list = [ACCENT, GREEN, PURPLE, ORANGE]
    
    for gi, (group_name, (col, vals)) in enumerate(groups.items()):
        color = colors_list[gi % len(colors_list)]
        for val in vals:
            mask = df[col] == val
            f1_vals = df.loc[mask, 'best_val_f1'].values
            if len(f1_vals) > 0:
                bp = ax.boxplot([f1_vals], positions=[pos], widths=0.5,
                                patch_artist=True, showfliers=True,
                                boxprops=dict(facecolor=color, alpha=0.4, edgecolor=color),
                                whiskerprops=dict(color=color),
                                capprops=dict(color=color),
                                medianprops=dict(color=GOLD, linewidth=2),
                                flierprops=dict(markeredgecolor=color, markersize=4))
                # Overlay individual points
                jitter = np.random.uniform(-0.12, 0.12, size=len(f1_vals))
                ax.scatter([pos] * len(f1_vals) + jitter, f1_vals, 
                          color=color, s=25, alpha=0.7, zorder=3, edgecolors='white', linewidths=0.3)
            labels.append(f'{val}')
            pos += 1
        pos += 0.5  # gap between groups
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5, rotation=30, ha='right')
    ax.set_ylabel('Best Validation F1 (%)')
    ax.set_title('F1 by Hyperparameter Choice', fontweight='bold', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add group separators and labels
    pos = 0
    for gi, (group_name, (col, vals)) in enumerate(groups.items()):
        mid = pos + len(vals) / 2 - 0.5
        ax.text(mid, ax.get_ylim()[1] + 0.05, group_name, ha='center', fontsize=8.5,
                color=colors_list[gi % len(colors_list)], fontweight='bold')
        pos += len(vals) + 0.5


def main():
    parser = argparse.ArgumentParser(description='Visualize experiment configurations')
    parser.add_argument('--csv', type=str, default='experiments.csv',
                        help='Path to experiments CSV file')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save figure (e.g., results/experiments_overview.png)')
    args = parser.parse_args()
    
    df = load_data(args.csv)
    print(f"Loaded {len(df)} experiments from {args.csv}")
    
    # ── Create figure ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle('Experiment Configurations Overview', fontsize=16, fontweight='bold',
                 color='#f0f6fc', y=0.98)
    
    # Layout: 2x2 grid with left panel taller
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                          left=0.06, right=0.97, top=0.93, bottom=0.06)
    
    ax1 = fig.add_subplot(gs[:, 0])   # Full left: F1 bars
    ax2 = fig.add_subplot(gs[0, 1])   # Top right: F1 vs params
    ax3 = fig.add_subplot(gs[1, 1])   # Bottom right: grouped F1
    
    plot_f1_bars(ax1, df)
    plot_f1_vs_params(ax2, df)
    plot_grouped_f1(ax3, df)
    
    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        fig.savefig(args.save, dpi=200, bbox_inches='tight')
        print(f"Saved to {args.save}")
    
    plt.show()
    
    # ── Also create heatmap as separate figure (better readability) ──────
    fig2, ax_hm = plt.subplots(figsize=(20, 6))
    fig2.patch.set_facecolor('#0d1117')
    plot_heatmap(ax_hm, df)
    fig2.suptitle('Hyperparameter Configuration Heatmap', fontsize=14, 
                  fontweight='bold', color='#f0f6fc')
    fig2.tight_layout()
    
    if args.save:
        hm_path = args.save.replace('.png', '_heatmap.png')
        fig2.savefig(hm_path, dpi=200, bbox_inches='tight')
        print(f"Saved heatmap to {hm_path}")
    
    plt.show()


if __name__ == '__main__':
    main()
