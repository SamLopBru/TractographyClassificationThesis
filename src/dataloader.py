from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
from typing import List
import h5py
import numpy as np  


class BalancedTractDataset(Dataset):
    def __init__(self, hdf5_file_paths, sampling_percentage=0.20, min_streamlines=50, 
                 max_streamlines=None, tract_ids=None, keep_padded=True):
        """
        Args:
            hdf5_file_paths: List of HDF5 file paths
            sampling_percentage: Percentage of streamlines to sample from each tract (0-1)
            min_streamlines: Minimum number of streamlines per tract
            max_streamlines: Maximum number of streamlines per tract (None = no limit)
            tract_ids: List of specific tract IDs to include (None = all tracts)
            keep_padded: If True, returns padded arrays (much faster)
        """
        self.file_paths = hdf5_file_paths
        self.sampling_percentage = sampling_percentage
        self.min_streamlines = min_streamlines
        self.max_streamlines = max_streamlines
        self.target_tract_ids = tract_ids
        self.keep_padded = keep_padded
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        with h5py.File(self.file_paths[idx], 'r', rdcc_nbytes=1024**3, rdcc_nslots=10000) as f:
            sampled_data = []
            
            for group_name in f.keys():
                if not group_name.startswith('tract_'):
                    continue
                
                tract_group = f[group_name]
                tract_id = tract_group.attrs['tract_id']
                
                if self.target_tract_ids and tract_id not in self.target_tract_ids:
                    continue
                
                n_available = tract_group.attrs['n_streamlines']
                n_percentage = int(n_available * self.sampling_percentage)
                n_to_sample = max(self.min_streamlines, n_percentage)
                
                if self.max_streamlines is not None:
                    n_to_sample = min(n_to_sample, self.max_streamlines)
                
                n_to_sample = min(n_to_sample, n_available)
                
                if n_to_sample < n_available:
                    idx_sample = np.sort(np.random.choice(n_available, n_to_sample, replace=False))
                    sampled_streamlines_padded = tract_group['streamlines'][idx_sample]
                    sampled_lengths = tract_group['lengths'][idx_sample]
                else:
                    sampled_streamlines_padded = tract_group['streamlines'][:]
                    sampled_lengths = tract_group['lengths'][:]
                
                if self.keep_padded:
                    # Much faster - return padded arrays directly
                    sampled_data.append({
                        'streamlines': torch.from_numpy(sampled_streamlines_padded),
                        'lengths': torch.from_numpy(sampled_lengths),
                        'tract_id': tract_id,
                        'n_streamlines': len(sampled_streamlines_padded)
                    })
                else:
                    # Optimized unpadding using list comprehension
                    unpadded_streamlines = [
                        torch.from_numpy(sampled_streamlines_padded[i, :sampled_lengths[i]].copy())
                        for i in range(len(sampled_lengths))
                    ]
                    
                    sampled_data.append({
                        'streamlines': unpadded_streamlines,
                        'tract_id': tract_id,
                        'n_streamlines': len(unpadded_streamlines)
                    })
        
        return sampled_data

def custom_collate(batch):
    """Custom collate that handles variable-length tract data"""
    # batch is a list of samples, where each sample is a list of tract dicts
    return batch  # Return as-is, don't try to stack


def verify_against_original():
    """Compare sampled data with original HDF5 file"""
    dataset = BalancedTractDataset(
        ["preprocessing/sequences/sub-1263.hdf5"],
        streamlines_per_tract=21113
    )
    
    print("="*50)
    print("ORIGINAL FILE STATS")
    print("="*50)
    
    with h5py.File("preprocessing/sequences/sub-1263.hdf5", 'r') as f:
        for group_name in f.keys():
            if not group_name.startswith('tract_'):
                continue
            
            tract_group = f[group_name]
            tract_id = tract_group.attrs['tract_id']
            n_streamlines = tract_group.attrs['n_streamlines']
            lengths = tract_group['lengths'][:]
            
            print(f"\nTract {tract_id}:")
            print(f"  Total streamlines in file: {n_streamlines}")
            print(f"  Length range: {lengths.min()} - {lengths.max()}")
            print(f"  Mean length: {lengths.mean():.2f}")
    
    print("\n" + "="*50)
    print("SAMPLED DATA STATS")
    print("="*50)
    
    sample = dataset[0]
    for tract_dict in sample['tract_data']:
        streamlines = tract_dict['streamlines']
        lengths = [s.shape[0] for s in streamlines]
        
        print(f"\nTract {tract_dict['tract_id']}:")
        print(f"  Sampled streamlines: {tract_dict['n_streamlines']}")
        print(f"  Length range: {min(lengths)} - {max(lengths)}")
        print(f"  Mean length: {np.mean(lengths):.2f}")
        print(f"  First streamline shape: {streamlines[0].shape}")

if __name__ == "__main__":
    import time
    import os
    
    paths = [os.path.join("preprocessing/sequences",path) for path in os.listdir("preprocessing/sequences")]
    dataset = BalancedTractDataset(
        paths,
        sampling_percentage=0.01,
        keep_padded=True
    )

    dataloader_fast = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=custom_collate  
    )

    start = time.time()
    for batch in dataloader_fast:
        print(f"Batch size: {batch[0]}")
        break
    
    print(f"With workers: {time.time() - start:.2f} seconds")
    