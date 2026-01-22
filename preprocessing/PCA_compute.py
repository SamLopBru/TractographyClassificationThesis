import numpy as np
import sklearn.decomposition as PCA
import nibabel as nib
from pathlib import Path
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))
from dataset_handler import Tractoinferno_handler


DUMMY_PATH = "/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives/testset/sub-1006/anat/sub-1006__T1w.nii.gz"
DUMMY_STREAMLINE = Path("/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives/testset/sub-1006/tractography/sub-1006__CC_Pa.trk")


def compute_principal_components(nib_path: Path, com: np.ndarray):

    img = nib.load(nib_path)
    img_data = img.get_fdata()
    mask = img_data > 0
    
    coords = np.argwhere(mask)
    coords_centered = coords - com

    pca = PCA.PCA(n_components=3)
    pca.fit(coords_centered)
    
    pca_axes = pca.components_
    #principal_components = pca.transform(coords_centered) 

    return pca_axes, pca.explained_variance_ratio_

def plot_overlaid_brains_with_pcs(nib_paths: list, coms: list, pca_axes_list: list, scale=50):
    """
    Plot all brain MRIs overlaid with their principal component axes
    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    brain_colors = plt.cm.tab10(np.linspace(0, 1, len(nib_paths)))
    
    for idx, (nib_path, com, pca_axes, brain_color) in enumerate(zip(nib_paths, coms, pca_axes_list, brain_colors)):
        img = nib.load(nib_path)
        img_data = img.get_fdata()
        mask = img_data > 0
        coords = np.argwhere(mask)
        
        # Plot center of mass
        ax.scatter(*com, c=[brain_color], s=100, marker='o', edgecolors='black')
        
        # Plot principal component axes
        pc_colors = ['red', 'green', 'blue']
        
        for i, (pc, color) in enumerate(zip(pca_axes, pc_colors)):
            ax.quiver(com[0], com[1], com[2],
                     pc[0]*scale, pc[1]*scale, pc[2]*scale,
                     color=color, arrow_length_ratio=0.15, linewidth=2,
                     alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('All Brains with Principal Components (Overlaid)')
    ax.legend()
    ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    return fig

def compare_variance_explained(variance_ratios: list):
    """
    Compare variance explained by each PC across multiple brains
    """
    
    variance_ratios = np.array(variance_ratios)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual variance ratios
    x = np.arange(len(nib_paths))
    width = 0.25
    for i in range(3):
        ax1.bar(x + i*width, variance_ratios[:, i], width, 
                label=f'PC{i+1}', alpha=0.8)
    
    ax1.set_xlabel('Brain')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Variance Explained by Each PC')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f'Brain {i+1}' for i in range(len(nib_paths))])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Statistics
    mean_var = variance_ratios.mean(axis=0)
    std_var = variance_ratios.std(axis=0)
    
    ax2.bar(['PC1', 'PC2', 'PC3'], mean_var, yerr=std_var, 
            capsize=5, alpha=0.8, color=['red', 'green', 'blue'])
    ax2.set_ylabel('Mean Explained Variance Ratio')
    ax2.set_title('Mean ± Std Across All Brains')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Print statistics
    print("Variance Explained Statistics:")
    print(f"Mean: PC1={mean_var[0]:.3f}, PC2={mean_var[1]:.3f}, PC3={mean_var[2]:.3f}")
    print(f"Std:  PC1={std_var[0]:.3f}, PC2={std_var[1]:.3f}, PC3={std_var[2]:.3f}")
    print(f"\nCoefficient of Variation (lower is better):")
    print(f"PC1={std_var[0]/mean_var[0]:.3f}, PC2={std_var[1]/mean_var[1]:.3f}, PC3={std_var[2]/mean_var[2]:.3f}")
    
    return variance_ratios, fig

def compare_pc_alignment(pca_axes_list):
    """
    Calculate angular differences between PC axes across brains
    """
    n_brains = len(pca_axes_list)
    
    # Compute pairwise angles between corresponding PCs
    angles = np.zeros((3, n_brains, n_brains))  # [PC_index, brain_i, brain_j]
    
    for pc_idx in range(3):
        for i in range(n_brains):
            for j in range(i+1, n_brains):
                # Get the two PC axes
                pc_i = pca_axes_list[i][pc_idx]
                pc_j = pca_axes_list[j][pc_idx]
                
                # Calculate angle (use absolute value to account for sign flip)
                cos_angle = np.abs(np.dot(pc_i, pc_j) / 
                                   (np.linalg.norm(pc_i) * np.linalg.norm(pc_j)))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_deg = np.arccos(cos_angle) * 180 / np.pi
                
                angles[pc_idx, i, j] = angle_deg
                angles[pc_idx, j, i] = angle_deg
    
    # Plot heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    for pc_idx, ax in enumerate(axes):
        im = ax.imshow(angles[pc_idx], cmap='RdYlGn_r', vmin=0, vmax=90)
        ax.set_title(f'PC{pc_idx+1} Angular Differences (degrees)')
        ax.set_xlabel('Brain')
        ax.set_ylabel('Brain')
        plt.colorbar(im, ax=ax)

        # Add text annotations
        for i in range(n_brains):
            for j in range(n_brains):
                if i != j:
                    text = ax.text(j, i, f'{angles[pc_idx, i, j]:.1f}',
                                 ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    
    # Print statistics
    print("\nPC Alignment Statistics (Angular Differences):")
    for pc_idx in range(3):
        # Get upper triangle (unique pairs)
        upper_tri = angles[pc_idx][np.triu_indices(n_brains, k=1)]
        print(f"\nPC{pc_idx+1}:")
        print(f"  Mean angle: {upper_tri.mean():.2f}°")
        print(f"  Max angle:  {upper_tri.max():.2f}°")
        print(f"  Std angle:  {upper_tri.std():.2f}°")
    
    return angles

def assess_alignment_quality(angles):
    """
    Provide recommendations based on alignment
    """
    print("\n" + "="*60)
    print("ASSESSMENT: Suitability for Relative Coordinates")
    print("="*60)
    
    for pc_idx in range(3):
        upper_tri = angles[pc_idx][np.triu_indices(angles.shape[1], k=1)]
        mean_angle = upper_tri.mean()
        max_angle = upper_tri.max()
        
        print(f"\nPC{pc_idx+1}:")
        if mean_angle < 10 and max_angle < 20:
            print(f"  ✓ EXCELLENT alignment (mean={mean_angle:.1f}°, max={max_angle:.1f}°)")
            print(f"    Highly consistent - suitable for relative coordinates")
        elif mean_angle < 20 and max_angle < 30:
            print(f"  ✓ GOOD alignment (mean={mean_angle:.1f}°, max={max_angle:.1f}°)")
            print(f"    Reasonably consistent - usable with caution")
        elif mean_angle < 30 and max_angle < 45:
            print(f"  ⚠ MODERATE alignment (mean={mean_angle:.1f}°, max={max_angle:.1f}°)")
            print(f"    Inconsistent - not recommended for relative coordinates")
        else:
            print(f"  ✗ POOR alignment (mean={mean_angle:.1f}°, max={max_angle:.1f}°)")
            print(f"    Highly inconsistent - do NOT use for relative coordinates")

def compute_consistency_score(variance_ratios, angles):
    """
    Compute overall consistency score (0-100)
    """
    # Variance consistency (lower CV is better)
    var_cv = variance_ratios.std(axis=0) / variance_ratios.mean(axis=0)
    var_score = np.mean(np.exp(-var_cv * 2)) * 100  # Convert to 0-100
    
    # Angular consistency (lower angles are better)
    angle_scores = []
    for pc_idx in range(3):
        upper_tri = angles[pc_idx][np.triu_indices(angles.shape[1], k=1)]
        mean_angle = upper_tri.mean()
        # Score decreases with angle (90° = 0 score, 0° = 100 score)
        angle_scores.append(max(0, 100 - mean_angle * 1.5))
    
    angle_score = np.mean(angle_scores)
    
    # Combined score
    overall_score = 0.4 * var_score + 0.6 * angle_score
    
    print(f"\n" + "="*60)
    print("CONSISTENCY SCORES")
    print("="*60)
    print(f"Variance Consistency Score: {var_score:.1f}/100")
    print(f"Angular Consistency Score:  {angle_score:.1f}/100")
    print(f"Overall Consistency Score:  {overall_score:.1f}/100")
    print()
    
    if overall_score >= 80:
        print("✓ RECOMMENDATION: PCs are highly consistent - EXCELLENT for relative coordinates")
    elif overall_score >= 60:
        print("✓ RECOMMENDATION: PCs are moderately consistent - ACCEPTABLE for relative coordinates")
    elif overall_score >= 40:
        print("⚠ RECOMMENDATION: PCs show some inconsistency - use with CAUTION")
    else:
        print("✗ RECOMMENDATION: PCs are inconsistent - NOT RECOMMENDED for relative coordinates")
    
    return overall_score

def load_subject_data(csv, subject: str) -> Tuple[np.ndarray, float, float]:
    subject_data = csv[csv['subject'] == subject].iloc[0]
    return (subject_data['com_x'], subject_data['com_y'], subject_data['com_z']), subject_data['r_min'], subject_data['r_max']

def identify_outlier_brains(angles, nib_paths, threshold=30):
    """
    Find which brains are poorly aligned
    """
    n_brains = len(nib_paths)
    
    # For each brain, calculate mean angle to all others
    mean_angles_per_brain = np.zeros((3, n_brains))
    
    for pc_idx in range(3):
        for i in range(n_brains):
            # Mean angle from brain i to all other brains
            angles_i = angles[pc_idx, i, :]
            mean_angles_per_brain[pc_idx, i] = angles_i[angles_i > 0].mean()
    
    print("Mean Angular Deviation per Brain:")
    print("="*60)
    
    outliers = []
    for i in range(n_brains):
        avg_deviation = mean_angles_per_brain[:, i].mean()
        status = "⚠ OUTLIER" if avg_deviation > threshold else "✓ Normal"
        print(f"Brain {i+1} ({nib_paths[i].name}): {avg_deviation:.2f}° {status}")
        
        if avg_deviation > threshold:
            outliers.append(i)
    
    print(f"\n{len(outliers)} outlier brain(s) detected")
    
    # Show detailed angles for outliers
    if outliers:
        print("\nDetailed Outlier Analysis:")
        for outlier_idx in outliers:
            print(f"\nBrain {outlier_idx+1}:")
            for pc_idx in range(3):
                print(f"  PC{pc_idx+1}: {mean_angles_per_brain[pc_idx, outlier_idx]:.2f}°")
    
    return outliers, mean_angles_per_brain

def plot_brain_deviation(mean_angles_per_brain, nib_paths):
    """
    Plot which brains deviate most from consensus
    """
    avg_deviation = mean_angles_per_brain.mean(axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['red' if d > 30 else 'green' if d < 15 else 'orange' 
              for d in avg_deviation]
    
    bars = ax.bar(range(len(nib_paths)), avg_deviation, color=colors, alpha=0.7)
    ax.axhline(y=15, color='green', linestyle='--', label='Excellent (<15°)')
    ax.axhline(y=30, color='red', linestyle='--', label='Outlier (>30°)')
    
    ax.set_xlabel('Brain')
    ax.set_ylabel('Mean Angular Deviation (degrees)')
    ax.set_title('Brain Alignment Deviation from Group Average')
    ax.set_xticks(range(len(nib_paths)))
    ax.set_xticklabels([f'Brain {i+1}' for i in range(len(nib_paths))], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def identify_problematic_pairs(angles, nib_paths, threshold=30):
    """
    Find which specific pairs of brains are poorly aligned
    """
    n_brains = len(nib_paths)
    
    print("Problematic Brain Pairs (Angular Difference > {}°):".format(threshold))
    print("="*60)
    
    problematic_pairs = []
    
    for pc_idx in range(3):
        print(f"\nPC{pc_idx+1}:")
        pairs_found = False
        
        for i in range(n_brains):
            for j in range(i+1, n_brains):
                angle = angles[pc_idx, i, j]
                
                if angle > threshold:
                    print(f"  Brain {i+1} ↔ Brain {j+1}: {angle:.2f}°")
                    problematic_pairs.append((i, j, pc_idx, angle))
                    pairs_found = True
        
        if not pairs_found:
            print(f"  None (all pairs < {threshold}°)")
    
    print(f"\n{len(problematic_pairs)} problematic pair(s) found across all PCs")
    
    return problematic_pairs

def analyze_pairwise_structure(angles, nib_paths):
    """
    Visualize the pairwise structure to identify subgroups
    """
    n_brains = len(nib_paths)
    
    # Average angle across all 3 PCs for each pair
    avg_angles = angles.mean(axis=0)
    
    # Create enhanced heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap
    im1 = ax1.imshow(avg_angles, cmap='RdYlGn_r', vmin=0, vmax=50)
    ax1.set_title('Average Angular Difference Across All PCs', fontsize=14)
    ax1.set_xlabel('Brain')
    ax1.set_ylabel('Brain')
    
    # Add values
    for i in range(n_brains):
        for j in range(n_brains):
            if i != j:
                color = 'white' if avg_angles[i, j] > 30 else 'black'
                text = ax1.text(j, i, f'{avg_angles[i, j]:.1f}',
                              ha="center", va="center", color=color, fontsize=9)
    
    plt.colorbar(im1, ax=ax1, label='Angle (degrees)')
    
    # Distribution of pairwise angles
    upper_tri = avg_angles[np.triu_indices(n_brains, k=1)]
    
    ax2.hist(upper_tri, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(upper_tri.mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {upper_tri.mean():.1f}°')
    ax2.axvline(30, color='orange', linestyle='--', 
                linewidth=2, label='Threshold: 30°')
    ax2.set_xlabel('Angular Difference (degrees)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Pairwise Angular Differences')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Statistics
    print("\nPairwise Angle Distribution:")
    print("="*60)
    print(f"Median: {np.median(upper_tri):.2f}°")
    print(f"Mean:   {upper_tri.mean():.2f}°")
    print(f"Std:    {upper_tri.std():.2f}°")
    print(f"Q1-Q3:  {np.percentile(upper_tri, 25):.2f}° - {np.percentile(upper_tri, 75):.2f}°")
    print(f"Min-Max: {upper_tri.min():.2f}° - {upper_tri.max():.2f}°")
    print(f"\nPairs with angle > 30°: {np.sum(upper_tri > 30)} / {len(upper_tri)} ({100*np.sum(upper_tri > 30)/len(upper_tri):.1f}%)")
    print(f"Pairs with angle > 40°: {np.sum(upper_tri > 40)} / {len(upper_tri)} ({100*np.sum(upper_tri > 40)/len(upper_tri):.1f}%)")
    
    return fig, upper_tri

if __name__ == "__main__":
    csv_path = Path("preprocessing/csvs/normalization_parameters_trainset.csv")
    coms = []
    nib_paths = []
    pca_axes_list = []
    variance_ratios = []

    dataset_handler = Tractoinferno_handler("/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives", scope="trainset")

    for subject in dataset_handler.get_data():
        nib_path = subject["T1w"]
        csv = pd.read_csv(csv_path)
        com, r_min, r_max = load_subject_data(csv, subject["subject"])
        nib_paths.append(nib_path)
        coms.append(com)
        pca_axes, variance_ratio = compute_principal_components(nib_path, com)
        pca_axes_list.append(pca_axes)
        variance_ratios.append(variance_ratio)


    # Plot
    fig = plot_overlaid_brains_with_pcs(nib_paths, coms, pca_axes_list, scale=50)
    plt.show()

    # Optional: Save figure
    fig.savefig('images/PCAs/brain_pcs.png', dpi=300, bbox_inches='tight')

    print("ANALYZING PCA CONSISTENCY...")
    print("="*60)
    
    variance_ratios, fig1 = compare_variance_explained(variance_ratios)
    angles = compare_pc_alignment(pca_axes_list)
    assess_alignment_quality(angles)
    overall_score = compute_consistency_score(variance_ratios, angles)
    outliers, mean_angles_per_brain = identify_outlier_brains(angles, nib_paths)
    fig3 = plot_brain_deviation(mean_angles_per_brain, nib_paths)
    problematic_pairs = identify_problematic_pairs(angles, nib_paths)
    fig4, _ = analyze_pairwise_structure(angles, nib_paths)
    plt.show()
    
    print("="*60)
    print("PCA CONSISTENCY ANALYSIS")
    print("="*60)
    print(f"Consistency Score: {overall_score:.1f}/100")
    print("="*60)
    
    # print({
    #     'coms': coms,
    #     'pca_axes': pca_axes_list,
    #     'variance_ratios': variance_ratios,
    #     'angles': angles,
    #     'consistency_score': overall_score
    # })



    



