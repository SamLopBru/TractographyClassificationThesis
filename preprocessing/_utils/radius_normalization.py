import nibabel as nib
import numpy as np
from dipy.io.streamline import load_tractogram
from scipy.ndimage import center_of_mass
import time
import sys
import os
import pandas as pd
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from typing import Tuple
from utils.dataset_handler import Tractoinferno_handler

def compute_global_normalization_ranges(streamlines: list[np.ndarray], global_center: np.ndarray) -> Tuple[float, float]:
    """Compute global min and max for spherical radius across all streamlines"""
    
    all_points = np.vstack(streamlines)
    centered = all_points - global_center
    r = np.linalg.norm(centered, axis=1)
    
    return r.min(), r.max()

def compute_com(mri: nib.Nifti1Image) -> np.ndarray:
    """Compute center of mass of a Nifti image"""
    mri = nib.as_closest_canonical(mri)
    mri_data = mri.get_fdata()
    affine = mri.affine

    com_vox = center_of_mass(mri_data>0)
    com_ras = np.dot(affine, np.append(com_vox, 1))[:3]
        
    return com_ras


def main(scope: str):
    start_time = time.time()
    dataset_handler = Tractoinferno_handler("/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives", scope=scope)
    
    results = []
    
    for subject in dataset_handler.subjects:
        
        subject_data = dataset_handler.get_data_from_subject(subject)
        mri_path = subject_data["T1w"]
        mri = nib.load(mri_path)
        com = compute_com(mri)

        all_streamlines = [
            streamline 
            for tract_path in subject_data["tracts"]
            for streamline in load_tractogram(str(tract_path), str(mri_path)).streamlines
        ]

        r_min, r_max = compute_global_normalization_ranges(all_streamlines, com)
        
        # Store results
        results.append({
            'subject': subject_data["subject"],
            'com_x': com[0],
            'com_y': com[1],
            'com_z': com[2],
            'r_min': r_min,
            'r_max': r_max
        })
        
        print(f"Subject {subject}: r_min={r_min:.2f}, r_max={r_max:.2f}")
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(f'preprocessing/csvs/normalization_parameters_{scope}.csv', index=False)
    print(f"\nSaved results for {len(results)} subjects to preprocessing/csvs/normalization_parameters_{scope}.csv")
    
    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main("testset")
