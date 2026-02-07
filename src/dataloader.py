from torch.utils.data import Dataset, DataLoader, Sampler
import torch
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import List, Tuple, Optional, Iterator
import h5py
import numpy as np  


class EpochSubsetSampler(Sampler[int]):
    """
    Sampler that provides a different random subset of indices each epoch.
    
    Call set_epoch(epoch) at the start of each epoch to get a new random subset.
    The subset size is determined by sampling_percentage, min_samples, and max_samples.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        sampling_percentage: float = 0.20,
        min_samples: int = 1000,
        max_samples: Optional[int] = None,
        seed: int = 42,
        shuffle: bool = True
    ):
        """
        Args:
            dataset: The dataset to sample from
            sampling_percentage: Percentage of total streamlines to sample (0-1)
            min_samples: Minimum number of samples per epoch
            max_samples: Maximum number of samples per epoch (None = no limit)
            seed: Base random seed for reproducibility
            shuffle: Whether to shuffle the sampled indices
        """
        self.dataset = dataset
        self.sampling_percentage = sampling_percentage
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        
        # Calculate number of samples per epoch
        total_size = len(dataset)
        n_percentage = int(total_size * sampling_percentage)
        self.n_samples = max(self.min_samples, n_percentage)
        
        if self.max_samples is not None:
            self.n_samples = min(self.n_samples, self.max_samples)
        
        self.n_samples = min(self.n_samples, total_size)
        
        print(f"EpochSubsetSampler: {self.n_samples} samples per epoch "
              f"(from {total_size} total, {self.sampling_percentage*100:.1f}%)")
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch number to generate a different random subset.
        
        Call this at the start of each epoch before iterating.
        """
        self.epoch = epoch
    
    def __iter__(self) -> Iterator[int]:
        """Generate a random subset of indices for this epoch."""
        # Use epoch-dependent seed for reproducibility
        rng = np.random.default_rng(self.seed + self.epoch)
        
        # Sample indices without replacement
        total_size = len(self.dataset)
        indices = rng.choice(total_size, self.n_samples, replace=False)
        
        if self.shuffle:
            rng.shuffle(indices)
        
        return iter(indices.tolist())
    
    def __len__(self) -> int:
        return self.n_samples


class StratifiedEpochSampler(Sampler[int]):
    """
    Stratified sampler that samples proportionally from EACH class each epoch.
    
    Guarantees that each class gets exactly sampling_percentage of its indexed 
    streamlines each epoch, maintaining class proportions.
    
    Call set_epoch(epoch) at the start of each epoch to get a new random subset.
    """
    
    def __init__(
        self,
        dataset: 'StreamlineDataset',
        sampling_percentage: float = 0.10,
        min_samples_per_class: int = 10,
        full_sample_threshold: int = 1000,
        seed: int = 42,
        shuffle: bool = True
    ):
        """
        Args:
            dataset: StreamlineDataset with streamline_index containing 'tract_id'
            sampling_percentage: Percentage to sample from EACH class (0-1)
            min_samples_per_class: Minimum samples per class per epoch
            full_sample_threshold: If class has fewer than this many indexed, take all (100%)
            seed: Base random seed for reproducibility
            shuffle: Whether to shuffle the final indices
        """
        self.dataset = dataset
        self.sampling_percentage = sampling_percentage
        self.min_samples_per_class = min_samples_per_class
        self.full_sample_threshold = full_sample_threshold
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        
        # Build class-to-indices mapping using NumPy for efficiency
        # Works with both dict-based and NumPy structured array index
        self.class_indices = {}
        
        # Check if dataset uses NumPy structured array (new memory-efficient version)
        if hasattr(dataset.streamline_index, 'dtype'):
            # NumPy structured array version
            tract_ids = dataset.streamline_index['tract_id']
            for idx in range(len(dataset)):
                tract_id = int(tract_ids[idx])
                if tract_id not in self.class_indices:
                    self.class_indices[tract_id] = []
                self.class_indices[tract_id].append(idx)
        else:
            # Legacy dict-based version (backward compatibility)
            for idx, item in enumerate(dataset.streamline_index):
                tract_id = item['tract_id']
                if tract_id not in self.class_indices:
                    self.class_indices[tract_id] = []
                self.class_indices[tract_id].append(idx)
        
        # Calculate samples per class
        self.samples_per_class = {}
        total_samples = 0
        for tract_id, indices in self.class_indices.items():
            n_available = len(indices)
            # If class has fewer than threshold, take all samples
            if n_available < self.full_sample_threshold:
                n_to_sample = n_available
            else:
                n_percentage = int(n_available * sampling_percentage)
                n_to_sample = max(self.min_samples_per_class, n_percentage)
                n_to_sample = min(n_to_sample, n_available)
            self.samples_per_class[tract_id] = n_to_sample
            total_samples += n_to_sample
        
        self.n_samples = total_samples
        
        print(f"StratifiedEpochSampler: {self.n_samples} samples per epoch "
              f"({self.sampling_percentage*100:.1f}% per class, full if <{self.full_sample_threshold})")
        # for tract_id in sorted(self.samples_per_class.keys()):
        #     print(f"  Class {tract_id}: {self.samples_per_class[tract_id]} / {len(self.class_indices[tract_id])}")
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch number to generate a different random subset.
        
        Call this at the start of each epoch before iterating.
        """
        self.epoch = epoch
    
    def __iter__(self) -> Iterator[int]:
        """Generate stratified random subset of indices for this epoch."""
        rng = np.random.default_rng(self.seed + self.epoch)
        
        all_indices = []
        
        # Sample from each class proportionally
        for tract_id, indices in self.class_indices.items():
            n_to_sample = self.samples_per_class[tract_id]
            sampled = rng.choice(indices, n_to_sample, replace=False)
            all_indices.extend(sampled.tolist())
        
        # Shuffle all indices together
        if self.shuffle:
            rng.shuffle(all_indices)
        
        return iter(all_indices)
    
    def __len__(self) -> int:
        return self.n_samples


class StreamlineDataset(Dataset):
    """
    Dataset that indexes a subset of streamlines from the HDF5 files.
    
    Each __getitem__ returns a single streamline with its label (tract_id).
    Use with EpochSubsetSampler to sample different subsets each epoch.
    
    Memory-optimized: Uses NumPy structured arrays instead of Python dicts
    to reduce RAM usage by ~90%.
    """
    
    # Structured array dtype for memory-efficient index
    INDEX_DTYPE = np.dtype([
        ('file_id', np.int16),        # Index into file_paths_lookup
        ('group_id', np.int16),       # tract_XX group number (0-31)
        ('streamline_idx', np.int32), # Index within the group
        ('length', np.int16),         # Streamline length
        ('tract_id', np.int8),        # Class label (0-31)
    ])
    
    def __init__(
        self, 
        hdf5_file_paths: List[str],
        sampling_percentage: float = 0.10,
        min_streamlines: int = 50,
        max_streamlines_per_tract: int = None,
        full_sample_threshold: int = 1000,
        seed: int = 42,
        cache_dir: str = None,
    ):
        """
        Args:
            hdf5_file_paths: List of HDF5 file paths
            sampling_percentage: Percentage of streamlines to index from each tract (0-1)
            min_streamlines: Minimum number of streamlines per tract
            max_streamlines_per_tract: Maximum number of streamlines per tract (None = no limit)
            full_sample_threshold: If tract has fewer than this many streamlines, take all (100%)
            seed: Random seed for reproducibility of the initial sampling
            cache_dir: Optional directory to cache the index (for faster startup)
        """
        self.file_paths = hdf5_file_paths
        self.sampling_percentage = sampling_percentage
        self.min_streamlines = min_streamlines
        self.max_streamlines_per_tract = max_streamlines_per_tract
        self.full_sample_threshold = full_sample_threshold
        self.seed = seed
        self.cache_dir = cache_dir
        
        # File path lookup table (maps file_id -> file_path)
        self.file_paths_lookup = list(hdf5_file_paths)
        self.file_path_to_id = {path: i for i, path in enumerate(self.file_paths_lookup)}
        
        # HDF5 file handle cache (opened lazily, per-worker for multiprocessing safety)
        self._file_handles = {}
        self._worker_id = None  # Track which worker owns these handles
        
        # Build index of sampled streamlines across all files
        # Stored as NumPy structured array for memory efficiency
        self.streamline_index = None  # Will be np.ndarray with INDEX_DTYPE
        self._build_index()
    
    def _build_index(self):
        """Build an index of sampled streamlines in the dataset."""
        print(f"Building streamline index (sampling {self.sampling_percentage*100:.1f}% per tract)...")
        
        rng = np.random.default_rng(self.seed)
        
        # Collect index entries as list first, then convert to NumPy
        index_entries = []
        
        for file_path in self.file_paths:
            file_id = self.file_path_to_id[file_path]
            
            with h5py.File(file_path, 'r') as f:
                for group_name in f.keys():
                    if not group_name.startswith('tract_'):
                        continue
                    
                    # Extract group number from 'tract_XX'
                    group_id = int(group_name.split('_')[1])
                    
                    tract_group = f[group_name]
                    tract_id = tract_group.attrs['tract_id']
                    n_available = tract_group.attrs['n_streamlines']
                    lengths = tract_group['lengths'][:]
                    
                    # Calculate how many streamlines to sample
                    if n_available < self.full_sample_threshold:
                        n_to_sample = n_available
                    else:
                        n_percentage = int(n_available * self.sampling_percentage)
                        n_to_sample = max(self.min_streamlines, n_percentage)
                    
                    if self.max_streamlines_per_tract is not None:
                        n_to_sample = min(n_to_sample, self.max_streamlines_per_tract)
                    
                    n_to_sample = min(n_to_sample, n_available)
                    
                    # Sample indices using seeded RNG
                    if n_to_sample < n_available:
                        sampled_indices = np.sort(rng.choice(n_available, n_to_sample, replace=False))
                    else:
                        sampled_indices = np.arange(n_available)
                    
                    # Add entries as tuples (will become structured array rows)
                    for idx in sampled_indices:
                        index_entries.append((
                            file_id,
                            group_id,
                            int(idx),
                            int(lengths[idx]),
                            int(tract_id)
                        ))
        
        # Convert to NumPy structured array (huge memory savings)
        self.streamline_index = np.array(index_entries, dtype=self.INDEX_DTYPE)
        
        print(f"Total streamlines indexed: {len(self.streamline_index)}")
        
        # Memory usage info
        memory_mb = self.streamline_index.nbytes / (1024 * 1024)
        print(f"  Index memory: {memory_mb:.1f} MB")
        
        # Count per class using NumPy (efficient)
        unique_classes, counts = np.unique(self.streamline_index['tract_id'], return_counts=True)
        print(f"  Classes: {len(unique_classes)}")
        print(f"  Streamlines per class: min={counts.min()}, max={counts.max()}")
    
    def __len__(self):
        return len(self.streamline_index)
    
    def _get_current_worker_id(self) -> int:
        """Get the current DataLoader worker ID, or -1 for main process."""
        worker_info = torch.utils.data.get_worker_info()
        return worker_info.id if worker_info is not None else -1
    
    def _get_file_handle(self, file_id: int):
        """Get a cached file handle, opening it if necessary.
        
        Handles are cached per-worker to be multiprocessing safe.
        """
        current_worker = self._get_current_worker_id()
        
        # If worker changed (shouldn't happen, but be safe), clear cache
        if self._worker_id is not None and self._worker_id != current_worker:
            self._close_file_handles()
        self._worker_id = current_worker
        
        # Open file if not cached
        if file_id not in self._file_handles:
            file_path = self.file_paths_lookup[file_id]
            self._file_handles[file_id] = h5py.File(file_path, 'r')
        
        return self._file_handles[file_id]
    
    def _close_file_handles(self):
        """Close all cached file handles."""
        for handle in self._file_handles.values():
            try:
                handle.close()
            except:
                pass
        self._file_handles = {}
    
    def __del__(self):
        """Cleanup file handles on deletion."""
        self._close_file_handles()
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get a single streamline.
        
        Returns:
            streamline: Tensor of shape (seq_len, 5)
            length: Actual length of the streamline
            tract_id: Label for the streamline
        """
        item = self.streamline_index[idx]
        
        # Get cached file handle (much faster than opening/closing each time)
        file_handle = self._get_file_handle(int(item['file_id']))
        group_name = f"tract_{item['group_id']}"
        
        tract_group = file_handle[group_name]
        
        # Get the single streamline
        streamline = tract_group['streamlines'][item['streamline_idx']]  # (max_len, 5)
        length = int(item['length'])
        
        # Trim to actual length (remove padding)
        streamline = streamline[:length]
        
        return (
            torch.from_numpy(streamline.astype(np.float32)),
            length,
            int(item['tract_id'])
        )
    
    def get_tract_id(self, idx: int) -> int:
        """Get tract_id for an index without loading the streamline (fast)."""
        return int(self.streamline_index[idx]['tract_id'])


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


# if __name__ == "__main__":
#     import time
#     import os
#     from collections import Counter
    
#     print("=" * 70)
#     print("STRATIFIED EPOCH SAMPLER VERIFICATION")
#     print("=" * 70)
    
#     paths = [os.path.join("sequences/testset", path) for path in os.listdir("sequences/testset")]
    
#     # Level 1: Dataset indexes 10% of all streamlines per tract
#     print("\n[1] Creating StreamlineDataset (indexing 10% per tract)...")
#     dataset = StreamlineDataset(
#         paths,
#         sampling_percentage=0.40,
#         min_streamlines=50,
#         seed=42
#     )
    
#     # Count indexed streamlines per class
#     indexed_per_class = Counter(item['tract_id'] for item in dataset.streamline_index)
#     print(f"\nIndexed streamlines per class:")
#     for tract_id in sorted(indexed_per_class.keys()):
#         print(f"  Class {tract_id}: {indexed_per_class[tract_id]}")
    
#     # Level 2: Stratified sampler provides proportional subset each epoch
#     print("\n[2] Creating StratifiedEpochSampler (10% per class per epoch)...")
#     sampler = StratifiedEpochSampler(
#         dataset,
#         sampling_percentage=0.25,
#         min_samples_per_class=10,
#         seed=42,
#         shuffle=True
#     )
    
#     # Verify stratified sampling across multiple epochs
#     print("\n" + "=" * 70)
#     print("VERIFICATION: Class distribution across epochs")
#     print("=" * 70)
    
#     for epoch in range(3):
#         sampler.set_epoch(epoch)
        
#         # Collect all indices for this epoch
#         epoch_indices = list(sampler)
        
#         # Count classes in this epoch's sample
#         epoch_class_counts = Counter(
#             dataset.streamline_index[idx]['tract_id'] for idx in epoch_indices
#         )
        
#         print(f"\n--- Epoch {epoch} ---")
#         print(f"Total samples: {len(epoch_indices)}")
#         print(f"Class distribution:")
        
#         for tract_id in sorted(epoch_class_counts.keys()):
#             count = epoch_class_counts[tract_id]
#             indexed = indexed_per_class[tract_id]
#             pct = 100.0 * count / indexed if indexed > 0 else 0
#             expected = sampler.samples_per_class[tract_id]
#             status = "✓" if count == expected else "✗"
#             print(f"  Class {tract_id}: {count:5d} / {indexed:5d} indexed "
#                   f"({pct:5.1f}%) [expected: {expected}] {status}")
    
#     # Verify different samples each epoch
#     print("\n" + "=" * 70)
#     print("VERIFICATION: Different samples each epoch")
#     print("=" * 70)
    
#     sampler.set_epoch(0)
#     epoch0_indices = set(sampler)
    
#     sampler.set_epoch(1)
#     epoch1_indices = set(sampler)
    
#     sampler.set_epoch(2)
#     epoch2_indices = set(sampler)
    
#     overlap_01 = len(epoch0_indices & epoch1_indices)
#     overlap_02 = len(epoch0_indices & epoch2_indices)
#     overlap_12 = len(epoch1_indices & epoch2_indices)
    
#     print(f"Epoch 0 samples: {len(epoch0_indices)}")
#     print(f"Epoch 1 samples: {len(epoch1_indices)}")
#     print(f"Epoch 2 samples: {len(epoch2_indices)}")
#     print(f"Overlap epoch 0-1: {overlap_01} ({100*overlap_01/len(epoch0_indices):.1f}%)")
#     print(f"Overlap epoch 0-2: {overlap_02} ({100*overlap_02/len(epoch0_indices):.1f}%)")
#     print(f"Overlap epoch 1-2: {overlap_12} ({100*overlap_12/len(epoch1_indices):.1f}%)")
    
#     # Quick DataLoader test
#     print("\n" + "=" * 70)
#     print("DATALOADER TEST")
#     print("=" * 70)
    
#     dataloader = DataLoader(
#         dataset,
#         batch_size=64,
#         shuffle=False,
#         sampler=sampler,
#         num_workers=4,
#         pin_memory=True,
#         collate_fn=streamline_collate_fn
#     )
    
#     sampler.set_epoch(0)
#     start = time.time()
#     batch_count = 0
#     sample_count = 0
#     batch_class_counts = Counter()
    
#     for streamlines, lengths, labels in dataloader:
#         batch_count += 1
#         sample_count += len(labels)
#         batch_class_counts.update(labels.tolist())
    
#     print(f"Processed {batch_count} batches, {sample_count} samples in {time.time() - start:.2f}s")
#     print(f"Class distribution in batches matches expected: ", end="")
    
#     all_match = all(
#         batch_class_counts[tid] == sampler.samples_per_class[tid]
#         for tid in sampler.samples_per_class
#     )
#     print("✓ YES" if all_match else "✗ NO")
    
#     print("\n" + "=" * 70)
#     print("VERIFICATION COMPLETE")
#     print("=" * 70)