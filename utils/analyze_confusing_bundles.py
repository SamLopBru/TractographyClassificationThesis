import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Fix python path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.dataloader import StreamlineDataset

def compute_streamline_metrics(streamlines):
    lengths = []
    euclidean_dists = []
    tortuosities = []
    
    all_points = []

    for sl in streamlines:
        sl = sl.numpy() if hasattr(sl, 'numpy') else sl
        all_points.append(sl)
        
        # Number of points
        num_points = sl.shape[0]
        lengths.append(num_points)
        
        # Euclidean distance
        if num_points > 1:
            dist = np.linalg.norm(sl[-1] - sl[0])
            euclidean_dists.append(dist)
            
            # Actual length
            diffs = sl[1:] - sl[:-1]
            actual_len = np.sum(np.linalg.norm(diffs, axis=1))
            
            # Tortuosity
            tort = actual_len / dist if dist > 0 else 1.0
            tortuosities.append(tort)
        else:
            euclidean_dists.append(0)
            tortuosities.append(1.0)
            
    if all_points:
        all_points_cat = np.concatenate(all_points, axis=0)
        min_coords = np.min(all_points_cat, axis=0)
        max_coords = np.max(all_points_cat, axis=0)
    else:
        min_coords = np.zeros(3)
        max_coords = np.zeros(3)

    return {
        'count': len(streamlines),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'mean_euc_dist': np.mean(euclidean_dists),
        'mean_tortuosity': np.mean(tortuosities),
        'bbox_min': min_coords,
        'bbox_max': max_coords
    }

def plot_bundles(bundle_dict, output_dir, samples_to_plot=50):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Individual plots
    for label, streamlines in bundle_dict.items():
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        plot_count = min(len(streamlines), samples_to_plot)
        sample_indices = np.random.choice(len(streamlines), plot_count, replace=False)
        
        for idx in sample_indices:
            sl = streamlines[idx]
            sl = sl.numpy() if hasattr(sl, 'numpy') else sl
            ax.plot(sl[:, 0], sl[:, 1], sl[:, 2], alpha=0.5, linewidth=1)
            
        ax.set_title(f'Bundle {label} ({plot_count} samples)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Consistent scale
        ax.set_box_aspect([1, 1, 1])
        
        out_path = os.path.join(output_dir, f'bundle_{label}_plot.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_path}")

    # 2. Combined plot
    fig = plt.figure(figsize=(15, 10))
    colors = {24: 'red', 25: 'blue', 26: 'green', 27: 'purple'}
    ax = fig.add_subplot(111, projection='3d')
    
    for label, streamlines in bundle_dict.items():
        plot_count = min(len(streamlines), int(samples_to_plot/2)) # Fewer per bundle in combined
        sample_indices = np.random.choice(len(streamlines), plot_count, replace=False)
        
        for idx in sample_indices:
            sl = streamlines[idx]
            sl = sl.numpy() if hasattr(sl, 'numpy') else sl
            ax.plot(sl[:, 0], sl[:, 1], sl[:, 2], color=colors[label], alpha=0.4, linewidth=1)
            
    # Proxy artists for legend
    import matplotlib.lines as mlines
    legend_elements = [mlines.Line2D([0], [0], color=c, lw=2, label=f'Bundle {lbl}') 
                       for lbl, c in colors.items() if lbl in bundle_dict]
    ax.legend(handles=legend_elements)
    
    ax.set_title('Combined View of Confusing Bundles (24, 25, 26, 27)')
    ax.set_box_aspect([1, 1, 1])
    
    combo_path = os.path.join(output_dir, 'combined_bundles_plot.png')
    plt.savefig(combo_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {combo_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir', type=str, default='sequences/testset')
    parser.add_argument('--max_samples', type=int, default=500, help='Max samples per bundle to extract')
    parser.add_argument('--out_dir', type=str, default='tests/bundle_analysis')
    args = parser.parse_args()

    # The bundles of interest
    target_bundles = [24, 25, 26, 27]
    
    print(f"Loading data from {args.val_dir}...")
    dataset = StreamlineDataset(
        [str(f) for f in Path(args.val_dir).glob('*.hdf5')],
        sampling_percentage=0.1,    # We only need a subset
        full_sample_threshold=1000   
    )
    
    bundle_data = {t: [] for t in target_bundles}
    found_counts = {t: 0 for t in target_bundles}
    
    print(f"Scanning dataset for bundles {target_bundles}...")
    
    # Randomly shuffle indices to get diverse samples
    indices = np.random.permutation(len(dataset))
    
    for idx in indices:
        tract_id = dataset.get_tract_id(idx)
        if tract_id in target_bundles and found_counts[tract_id] < args.max_samples:
            streamline, length, label = dataset[idx]
            bundle_data[tract_id].append(streamline[:length]) # Trim padded length if any
            found_counts[tract_id] += 1
            
        if all(c >= args.max_samples for c in found_counts.values()):
            print("Reached target capacity for all bundles.")
            break

    # Analyze and print stats
    print("\n--- BUNDLE STATISTICS ---")
    for b in target_bundles:
        if len(bundle_data[b]) > 0:
            stats = compute_streamline_metrics(bundle_data[b])
            print(f"\nBundle {b} ({stats['count']} samples):")
            print(f"  Length (pts):  Mean = {stats['mean_length']:.1f} ± {stats['std_length']:.1f} (range: {stats['min_length']}-{stats['max_length']})")
            print(f"  Euc Dist:      Mean = {stats['mean_euc_dist']:.3f}")
            print(f"  Tortuosity:    Mean = {stats['mean_tortuosity']:.3f}")
            print(f"  Bounding Box:  Min (x,y,z) = [{stats['bbox_min'][0]:.2f}, {stats['bbox_min'][1]:.2f}, {stats['bbox_min'][2]:.2f}]")
            print(f"                 Max (x,y,z) = [{stats['bbox_max'][0]:.2f}, {stats['bbox_max'][1]:.2f}, {stats['bbox_max'][2]:.2f}]")
        else:
            print(f"\nBundle {b}: No samples found!")

    # Generate plots
    print("\n--- GENERATING PLOTS ---")
    plot_bundles(bundle_data, args.out_dir)
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
