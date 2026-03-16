"""
CEUS Data Processor
SVD filtering is cached as separate files (memory efficient!)
ROI extraction happens after loading from cache
"""

import os
import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm

from data_loader import load_ceus_data
from filters import filter_both
from cache_utils import (
    get_cache_dir,
    save_svd_file,
    load_svd_file,
    save_cache_metadata,
    load_cache_metadata,
    validate_svd_cache,
    get_cache_size
)


def process_and_cache(data_dir, all_files, n_svd, cache_dir):
    """
    Process all files with SVD filtering
    Saves each file separately (memory efficient!)
    
    Args:
        data_dir: Directory containing .mat files
        all_files: List of filenames to process
        n_svd: Number of SVD components for tissue separation
        cache_dir: Directory to save cache files
        
    Returns:
        params dict with dt_ms
    """
    
    print("\nProcessing files with SVD filtering (one at a time)...")
    print("Saving each file separately to avoid memory issues")
    
    params = None
    
    for filename in tqdm(all_files, desc="SVD filtering", ncols=100):
        filepath = os.path.join(data_dir, filename)
        data = load_ceus_data(filepath)
        IQ = data['IQ']
        params = data['params']
        
        # Apply SVD filtering
        tissue, bubbles = filter_both(IQ, n_components=n_svd)
        
        # Save this file's result
        save_svd_file(cache_dir, filename, bubbles, tissue)
        
        # Free memory
        del IQ, tissue, bubbles
    
    # Save metadata
    save_cache_metadata(cache_dir, all_files, n_svd, params['dt_ms'])
    
    cache_size = get_cache_size(cache_dir)
    print(f"\n✓ SVD cache created!")
    print(f"  Location: {cache_dir}")
    print(f"  Size: {cache_size:.1f} MB")
    print(f"  Files: {len(all_files)}")
    
    return params


def load_all_svd_arrays(cache_dir, file_list):
    """
    Load all SVD arrays from cache
    
    Args:
        cache_dir: Cache directory
        file_list: List of filenames to load
        
    Returns:
        bubbles_arrays, tissue_arrays (lists of arrays)
    """
    print("\nLoading SVD arrays from cache...")
    
    bubbles_arrays = []
    tissue_arrays = []
    
    for filename in tqdm(file_list, desc="Loading cache", ncols=100):
        bubbles, tissue = load_svd_file(cache_dir, filename)
        
        if bubbles is None or tissue is None:
            raise ValueError(f"Cache file missing for {filename}")
        
        bubbles_arrays.append(bubbles)
        tissue_arrays.append(tissue)
    
    print(f"✓ Loaded {len(bubbles_arrays)} files from cache")
    
    return bubbles_arrays, tissue_arrays


def extract_roi_and_smooth(bubbles_arrays, tissue_arrays, dt_ms, roi_coords, smooth_window=301):
    """
    Extract ROI from full spatial arrays and apply smoothing
    
    Args:
        bubbles_arrays: List of full bubble arrays [(z,x,t), ...]
        tissue_arrays: List of full tissue arrays [(z,x,t), ...]
        dt_ms: Time step in milliseconds
        roi_coords: (z_min, z_max, x_min, x_max)
        smooth_window: Savitzky-Golay filter window size
        
    Returns:
        Dictionary with ROI time series and smoothed signals
    """
    
    z_min, z_max, x_min, x_max = roi_coords
    
    print(f"\n[Extracting ROI: z=[{z_min}:{z_max}], x=[{x_min}:{x_max}]]")
    
    # Extract ROI from each array and average spatially
    roi_bubbles_curves = []
    roi_tissue_curves = []
    
    for bubbles, tissue in zip(bubbles_arrays, tissue_arrays):
        # Extract ROI
        roi_bubbles = bubbles[z_min:z_max, x_min:x_max, :]
        roi_tissue = tissue[z_min:z_max, x_min:x_max, :]
        
        # Average over spatial dimensions
        roi_bubbles_curves.append(np.mean(np.abs(roi_bubbles), axis=(0, 1)))
        roi_tissue_curves.append(np.mean(np.abs(roi_tissue), axis=(0, 1)))
    
    # Concatenate all files
    bubbles_curve = np.concatenate(roi_bubbles_curves)
    tissue_curve = np.concatenate(roi_tissue_curves)
    
    # Time axis
    time_curve = np.arange(len(bubbles_curve)) * dt_ms / 1000.0
    
    print(f"  ROI size: {z_max-z_min} × {x_max-x_min} pixels")
    print(f"  Time series length: {len(bubbles_curve)} frames")
    
    # Apply smoothing
    print(f"\n[Smoothing (window={smooth_window})]")
    
    if smooth_window >= len(bubbles_curve):
        smooth_window = len(bubbles_curve) - 1
        if smooth_window % 2 == 0:
            smooth_window -= 1
        print(f"  Adjusted window to {smooth_window}")
    
    bubbles_smooth = savgol_filter(bubbles_curve, window_length=smooth_window, polyorder=3)
    tissue_smooth = savgol_filter(tissue_curve, window_length=smooth_window, polyorder=3)
    
    return {
        'time': time_curve,
        'bubbles': bubbles_curve,
        'tissue': tissue_curve,
        'bubbles_smooth': bubbles_smooth,
        'tissue_smooth': tissue_smooth
    }


def process_all_files(data_dir, n_svd, roi_coords, smooth_window=301,
                     use_cache=True, force_reprocess=False):
    """
    Process all CEUS files: SVD filtering (cached) + ROI extraction + smoothing
    
    Args:
        data_dir: Directory containing .mat files
        n_svd: Number of SVD components for tissue separation
        roi_coords: (z_min, z_max, x_min, x_max) - REQUIRED for extraction
        smooth_window: Savitzky-Golay filter window size
        use_cache: Use cached SVD results if available
        force_reprocess: Force SVD reprocessing even if cache exists
        
    Returns:
        Dictionary with processed data
    """
    
    print("="*70)
    print("PROCESSING CEUS FILES")
    print("="*70)
    
    # Validate ROI
    if roi_coords is None:
        raise ValueError("roi_coords is REQUIRED for ROI extraction!")
    
    z_min, z_max, x_min, x_max = roi_coords
    
    # Find files
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])
    print(f"\nFound {len(all_files)} files")
    
    if len(all_files) == 0:
        raise ValueError("No .mat files found!")
    
    # SVD Cache handling
    cache_dir = get_cache_dir(data_dir, n_svd)
    
    if use_cache and not force_reprocess:
        print("\n[1] Checking SVD cache...")
        is_valid, msg = validate_svd_cache(cache_dir, data_dir, n_svd)
        
        if is_valid:
            print(f"  ✓ {msg} - Loading from cache!")
            metadata = load_cache_metadata(cache_dir)
            dt_ms = metadata['dt_ms']
            
            # Load arrays from cache
            bubbles_arrays, tissue_arrays = load_all_svd_arrays(cache_dir, all_files)
        else:
            print(f"  ✗ {msg} - Running SVD filtering...")
            params = process_and_cache(data_dir, all_files, n_svd, cache_dir)
            dt_ms = params['dt_ms']
            
            # Load arrays we just created
            bubbles_arrays, tissue_arrays = load_all_svd_arrays(cache_dir, all_files)
    else:
        print("\n[1] Running SVD filtering...")
        params = process_and_cache(data_dir, all_files, n_svd, cache_dir if use_cache else None)
        dt_ms = params['dt_ms']
        
        # Load arrays we just created (or keep in memory if no cache)
        if use_cache:
            bubbles_arrays, tissue_arrays = load_all_svd_arrays(cache_dir, all_files)
    
    # ROI extraction and smoothing
    print("\n[2] Extracting ROI and smoothing...")
    roi_results = extract_roi_and_smooth(
        bubbles_arrays, tissue_arrays, dt_ms, roi_coords, smooth_window
    )
    
    duration = roi_results['time'][-1]
    print(f"\n✓ Processing complete!")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Frame rate: {1000.0/dt_ms:.1f} Hz")
    print("="*70)
    
    return {
        'time': roi_results['time'],
        'bubbles': roi_results['bubbles'],
        'tissue': roi_results['tissue'],
        'bubbles_smooth': roi_results['bubbles_smooth'],
        'tissue_smooth': roi_results['tissue_smooth'],
        'roi_coords': roi_coords,
        'n_files': len(all_files),
        'params': {'dt_ms': dt_ms}
    }