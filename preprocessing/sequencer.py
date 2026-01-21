import nibabel as nib
import numpy as np
from dipy.io.streamline import load_tractogram
from scipy.ndimage import center_of_mass
import time
from typing import Tuple, Dict, List
import pathlib
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils.dataset_handler import Tractoinferno_handler

DUMMY_PATH = "/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives/testset/sub-1006/anat/sub-1006__T1w.nii.gz"
DUMMY_STREAMLINE = pathlib.Path("/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives/testset/sub-1006/tractography/sub-1006__CC_Pa.trk")
ENCODED_TRACTS: Dict[str, int] = {'AF_L': 0, 'AF_R': 1, 'CC_Fr_1': 2, 'CC_Fr_2': 3, 'CC_Oc': 4, 'CC_Pa': 5, 'CC_Pr_Po': 6, 'CG_L': 7, 'CG_R': 8, 'FAT_L': 9, 'FAT_R': 10, 'FPT_L': 11, 'FPT_R': 12, 'FX_L': 13, 'FX_R': 14, 'IFOF_L': 15, 'IFOF_R': 16, 'ILF_L': 17, 'ILF_R': 18, 'MCP': 19, 'MdLF_L': 20, 'MdLF_R': 21, 'OR_ML_L': 22, 'OR_ML_R': 23, 'POPT_L': 24, 'POPT_R': 25, 'PYT_L': 26, 'PYT_R': 27, 'SLF_L': 28, 'SLF_R': 29, 'UF_L': 30, 'UF_R': 31}


class SphericalSequencer:
    def __init__(self, mri_path: str, csv_path: str = None, encoded_tracts: Dict[str, int] = None):
        self.mri_path = mri_path
        self.mri = nib.load(mri_path)

        self.encoded_tracts = encoded_tracts or ENCODED_TRACTS

        self.csv = pd.read_csv(csv_path)

    def load_subject_data(self, subject: str) -> Tuple[np.ndarray, float, float]:
        subject_data = self.csv[self.csv['subject'] == subject].iloc[0]
        return (subject_data['com_x'], subject_data['com_y'], subject_data['com_z']), subject_data['r_min'], subject_data['r_max']

    @staticmethod
    def _name_tract(tract: pathlib.Path) -> str:
        return tract.name.split("__")[-1].split(".")[0]
    
    @staticmethod
    def _process_single_streamline(streamline: np.ndarray, global_com: np.ndarray, r_min: float, r_max: float, epsilon: float = 1e-6) -> np.ndarray:
    
        centered_streamline = streamline - global_com
        
        # Radial distance - normalized to [0, 1]
        r = np.linalg.norm(centered_streamline, axis=1)
        r_normalized = (r - r_min) / (r_max - r_min + epsilon)
        
        # Theta (polar angle) - encode with sin/cos
        theta = np.arccos(np.clip(centered_streamline[:, 2] / (r + epsilon), -1, 1))
        theta_cos = np.cos(theta)
        theta_sin = np.sin(theta)
        
        # Phi (azimuthal angle) - encode with sin/cos
        phi = np.arctan2(centered_streamline[:, 1], centered_streamline[:, 0])
        phi_cos = np.cos(phi)
        phi_sin = np.sin(phi)
        
        # Return 5 features per point
        return np.stack((r_normalized, theta_cos, theta_sin, phi_cos, phi_sin), axis=1)

    def process_subject(
        self, 
        tractograms: List[pathlib.Path], 
        com: np.ndarray, 
        r_min: float, 
        r_max: float,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Process all tractograms and return streamlines with their tract IDs
        
        Returns:
            processed_streamlines: List of processed streamline arrays
            tract_ids: List of tract IDs corresponding to each streamline
        """
        processed_streamlines = []
        tract_ids = []

        for tract_path in tractograms:
            # Get tract name and encoded ID
            tract_name = self._name_tract(tract_path)
            tract_id = self.encoded_tracts.get(tract_name, -1)  # -1 if not found
            
            # Load and process streamlines
            streamlines = load_tractogram(str(tract_path), str(self.mri_path))
            
            for streamline in streamlines.streamlines:
                processed = self._process_single_streamline(streamline, com, r_min, r_max)
                processed_streamlines.append(processed)
                tract_ids.append(tract_id)

            del streamlines
        return processed_streamlines, tract_ids
    
    def process_and_save_subject(
        self,
        tractograms: List[pathlib.Path],
        subject: str,
        output_path: str = None
    ):
        """
        Process subject and save to compressed npz file
        
        Args:
            tractograms: List of tractogram file paths
            com: Center of mass
            r_min: Minimum radius for normalization
            r_max: Maximum radius for normalization
            output_path: Path to save the output npz file
        """

        com, r_min, r_max = self.load_subject_data(subject)

        processed_streamlines, tract_ids = self.process_subject(tractograms, com, r_min, r_max)
        
        # Save with metadata
        np.savez_compressed(
            output_path,
            streamlines=np.array(processed_streamlines, dtype=object),  # List of variable-length arrays
            tract_ids=np.array(tract_ids),
            com=np.array(com),
            r_min=r_min,
            r_max=r_max,
            n_streamlines=len(processed_streamlines),
            allow_pickle=True
        )
        print("Saved " + subject + " to " + output_path)

        del processed_streamlines, tract_ids


def main(scope: str):
    dataset_handler = Tractoinferno_handler("/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives", scope=scope)
    for subject in dataset_handler.get_data():
        sequencer = SphericalSequencer(mri_path=subject["T1w"], encoded_tracts=ENCODED_TRACTS, csv_path="preprocessing/csvs/normalization_parameters_testset.csv")
        sequencer.process_and_save_subject(subject["tracts"], subject["subject"], "preprocessing/sequences/" + subject["subject"] + ".npz")

if __name__ == "__main__":
    scope = "testset"
    main(scope)







    




