import numpy as np  
import h5py

def _load_sequences_concat(hdf5_path, indices=None):
    """Load from concatenated storage"""
    with h5py.File(hdf5_path, 'r') as f:
        com = f.attrs['com']
        r_min = f.attrs['r_min']
        r_max = f.attrs['r_max']
        
        all_points = f['points'][:]
        offsets = f['offsets'][:]
        tract_ids = f['tract_ids'][:]
        
        if indices is not None:
            # Load only requested streamlines
            streamlines = [all_points[offsets[i]:offsets[i+1]] for i in indices]
            tract_ids = tract_ids[indices]
        else:
            # Load all streamlines
            streamlines = [all_points[offsets[i]:offsets[i+1]] 
                          for i in range(len(offsets)-1)]
        
        return streamlines, tract_ids, com, r_min, r_max

def _spherical_to_cartesian(streamline, com, r_min, r_max):
    """
    Convert spherical sequence back to Cartesian coordinates
    
    Args:
        streamline: Array of shape (n_points, 5) with [r_norm, theta_cos, theta_sin, phi_cos, phi_sin]
        com: Center of mass to re-center the streamline
        r_min, r_max: Normalization parameters
    """
    # Extract features
    r_normalized = streamline[:, 0]
    theta_cos = streamline[:, 1]
    theta_sin = streamline[:, 2]
    phi_cos = streamline[:, 3]
    phi_sin = streamline[:, 4]
    
    # Denormalize radius
    r = r_normalized * (r_max - r_min) + r_min
    
    # Reconstruct angles
    theta = np.arctan2(theta_sin, theta_cos)
    phi = np.arctan2(phi_sin, phi_cos)
    
    # Convert to Cartesian
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # Re-center around COM
    cartesian = np.stack([x, y, z], axis=1) + com
    
    return cartesian
