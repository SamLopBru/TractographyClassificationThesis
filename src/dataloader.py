from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import List, Tuple
import h5py
import numpy as np  


class StreamlineDataset(Dataset):
    """
    Dataset that treats each streamline as an individual sample.
    
    Each __getitem__ returns a single streamline with its label (tract_id).
    This allows batching at the streamline level rather than subject level.
    """
    
    def __init__(
        self, 
        hdf5_file_paths: List[str],
        sampling_percentage: float = 0.20,
        min_streamlines: int = 50,
        max_streamlines_per_tract: int = None
    ):
        """
        Args:
            hdf5_file_paths: List of HDF5 file paths
            sampling_percentage: Percentage of streamlines to sample from each tract (0-1)
            min_streamlines: Minimum number of streamlines per tract
            max_streamlines_per_tract: Maximum number of streamlines per tract (None = no limit)
        """
        self.file_paths = hdf5_file_paths
        self.sampling_percentage = sampling_percentage
        self.min_streamlines = min_streamlines
        self.max_streamlines_per_tract = max_streamlines_per_tract
        
        # Build index of all streamlines across all files
        # Each entry: (file_path, tract_group_name, streamline_idx, length, tract_id)
        self.streamline_index = []
        self._build_index()
    
    def _build_index(self):
        """Build an index of all streamlines in the dataset."""
        print("Building streamline index...")
        
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as f:
                for group_name in f.keys():
                    if not group_name.startswith('tract_'):
                        continue
                    
                    tract_group = f[group_name]
                    tract_id = tract_group.attrs['tract_id']
                    n_available = tract_group.attrs['n_streamlines']
                    lengths = tract_group['lengths'][:]
                    
                    # Calculate how many streamlines to sample
                    n_percentage = int(n_available * self.sampling_percentage)
                    n_to_sample = max(self.min_streamlines, n_percentage)
                    
                    if self.max_streamlines_per_tract is not None:
                        n_to_sample = min(n_to_sample, self.max_streamlines_per_tract)
                    
                    n_to_sample = min(n_to_sample, n_available)
                    
                    # Sample indices
                    if n_to_sample < n_available:
                        sampled_indices = np.sort(np.random.choice(n_available, n_to_sample, replace=False))
                    else:
                        sampled_indices = np.arange(n_available)
                    
                    # Add each streamline to the index
                    for idx in sampled_indices:
                        self.streamline_index.append({
                            'file_path': file_path,
                            'group_name': group_name,
                            'streamline_idx': int(idx),
                            'length': int(lengths[idx]),
                            'tract_id': int(tract_id)
                        })
        
        print(f"  Total streamlines indexed: {len(self.streamline_index)}")
        
        # Count per class
        class_counts = {}
        for item in self.streamline_index:
            tid = item['tract_id']
            class_counts[tid] = class_counts.get(tid, 0) + 1
        print(f"  Classes: {len(class_counts)}")
        print(f"  Streamlines per class: min={min(class_counts.values())}, max={max(class_counts.values())}")
    
    def __len__(self):
        return len(self.streamline_index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get a single streamline.
        
        Returns:
            streamline: Tensor of shape (seq_len, 5)
            length: Actual length of the streamline
            tract_id: Label for the streamline
        """
        item = self.streamline_index[idx]
        
        with h5py.File(item['file_path'], 'r') as f:
            tract_group = f[item['group_name']]
            
            # Get the single streamline
            streamline = tract_group['streamlines'][item['streamline_idx']]  # (max_len, 5)
            length = item['length']
            
            # Trim to actual length (remove padding)
            streamline = streamline[:length]
        
        return (
            torch.from_numpy(streamline.astype(np.float32)),
            length,
            item['tract_id']
        )


def streamline_collate_fn(batch: List[Tuple[torch.Tensor, int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function that pads streamlines to the same length within a batch.
    
    Args:
        batch: List of (streamline, length, tract_id) tuples
    
    Returns:
        streamlines: Padded tensor of shape (batch_size, max_seq_len, 5)
        lengths: Tensor of shape (batch_size,) with actual lengths
        labels: Tensor of shape (batch_size,) with tract_ids
    """
    streamlines = [item[0] for item in batch]
    lengths = torch.tensor([item[1] for item in batch], dtype=torch.long)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    
    # Pad streamlines to max length in this batch
    # pad_sequence expects (seq_len, features) tensors and pads along dim 0
    padded_streamlines = pad_sequence(streamlines, batch_first=True, padding_value=0.0)
    
    return padded_streamlines, lengths, labels

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
    
    paths = [os.path.join("sequences/testset",path) for path in os.listdir("sequences/testset")]
    dataset = StreamlineDataset(
        paths,
        sampling_percentage=0.10,
    )

    dataloader_fast = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=streamline_collate_fn  
    )

    start = time.time()
    for batch in dataloader_fast:
        print(f"Batch size: {batch}")
        break
    

    