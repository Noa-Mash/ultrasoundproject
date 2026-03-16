"""
SVD Cache Utilities for CEUS Data Processing
Caches FULL SVD-filtered spatial arrays as SEPARATE FILES (memory efficient!)
This allows selecting different ROIs without re-running SVD
"""

import os
import numpy as np
import hashlib


CACHE_VERSION = 2


def get_cache_dir(data_dir, n_svd):
    """Get cache directory path"""
    dir_hash = hashlib.md5(data_dir.encode()).hexdigest()[:8]
    cache_dir_name = f"svd_cache_v{CACHE_VERSION}_{dir_hash}_n{n_svd}"
    return cache_dir_name


def save_svd_file(cache_dir, filename, bubbles, tissue):
    """
    Save SVD result for single file
    
    Args:
        cache_dir: Cache directory
        filename: Original filename (e.g. 'file_001.mat')
        bubbles: Bubble array (z, x, t)
        tissue: Tissue array (z, x, t)
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache filename
    base_name = os.path.splitext(filename)[0]
    cache_file = os.path.join(cache_dir, f"{base_name}_svd.npz")
    
    # Save
    np.savez_compressed(
        cache_file,
        bubbles=bubbles,
        tissue=tissue
    )


def load_svd_file(cache_dir, filename):
    """
    Load SVD result for single file
    
    Args:
        cache_dir: Cache directory
        filename: Original filename
        
    Returns:
        bubbles, tissue arrays
    """
    base_name = os.path.splitext(filename)[0]
    cache_file = os.path.join(cache_dir, f"{base_name}_svd.npz")
    
    if not os.path.exists(cache_file):
        return None, None
    
    data = np.load(cache_file)
    return data['bubbles'], data['tissue']


def save_cache_metadata(cache_dir, file_list, n_svd, dt_ms):
    """Save metadata about the cache"""
    metadata_file = os.path.join(cache_dir, "_metadata.npz")
    
    np.savez(
        metadata_file,
        version=CACHE_VERSION,
        file_list=file_list,
        n_svd=n_svd,
        dt_ms=dt_ms
    )


def load_cache_metadata(cache_dir):
    """Load cache metadata"""
    metadata_file = os.path.join(cache_dir, "_metadata.npz")
    
    if not os.path.exists(metadata_file):
        return None
    
    data = np.load(metadata_file, allow_pickle=True)
    return {
        'version': int(data['version']),
        'file_list': data['file_list'].tolist(),
        'n_svd': int(data['n_svd']),
        'dt_ms': float(data['dt_ms'])
    }


def validate_svd_cache(cache_dir, data_dir, n_svd):
    """
    Check if SVD cache is valid
    
    Args:
        cache_dir: Cache directory path
        data_dir: Data directory to check against
        n_svd: Expected number of SVD components
    
    Returns:
        (is_valid, message) tuple
    """
    if not os.path.exists(cache_dir):
        return False, "Cache directory not found"
    
    # Load metadata
    metadata = load_cache_metadata(cache_dir)
    if metadata is None:
        return False, "Cache metadata not found"
    
    # Check version
    if metadata['version'] != CACHE_VERSION:
        return False, f"Cache version mismatch (v{metadata['version']} vs v{CACHE_VERSION})"
    
    # Check n_svd
    if metadata['n_svd'] != n_svd:
        return False, f"n_svd mismatch (cached: {metadata['n_svd']}, requested: {n_svd})"
    
    # Check files
    current_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])
    cached_files = metadata['file_list']
    
    if len(current_files) != len(cached_files):
        return False, f"File count changed ({len(cached_files)} → {len(current_files)})"
    
    if current_files != cached_files:
        return False, "File list changed"
    
    # Check if all cache files exist
    for filename in cached_files:
        base_name = os.path.splitext(filename)[0]
        cache_file = os.path.join(cache_dir, f"{base_name}_svd.npz")
        if not os.path.exists(cache_file):
            return False, f"Missing cache file for {filename}"
    
    return True, "Valid"


def clear_cache(cache_dir):
    """Delete cache directory if it exists"""
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
        print(f"✓ Cache deleted: {cache_dir}")
        return True
    else:
        print(f"No cache to delete: {cache_dir}")
        return False


def get_cache_size(cache_dir):
    """Get total cache size in MB"""
    if not os.path.exists(cache_dir):
        return 0
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    
    return total_size / (1024 * 1024)