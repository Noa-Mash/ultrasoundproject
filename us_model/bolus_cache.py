"""
Bolus Cache Utilities for CEUS Data Processing
Caches lognormal bolus fitting results to avoid re-fitting
Follows the same design patterns as cache_utils.py (SVD cache)
"""

import os
import json
import shutil
import numpy as np
import hashlib


BOLUS_CACHE_VERSION = 1


def _compute_config_hash(data_dir, n_svd, roi_coords, smooth_window):
    """
    Compute a short hash from the bolus fitting configuration.

    This identifies a unique combination of data source, SVD filtering,
    ROI selection, and smoothing that produced the input time-series
    fed into the bolus fitter.
    """
    config_str = (
        f"{data_dir}|{n_svd}|{roi_coords}|{smooth_window}"
    )
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def get_bolus_cache_dir(data_dir, n_svd, roi_coords, smooth_window):
    """
    Get bolus cache directory path.

    Args:
        data_dir: Original data directory
        n_svd: Number of SVD components used upstream
        roi_coords: (z_min, z_max, x_min, x_max) ROI used
        smooth_window: Savitzky-Golay smoothing window size

    Returns:
        Cache directory name string
    """
    cfg_hash = _compute_config_hash(data_dir, n_svd, roi_coords, smooth_window)
    return f"bolus_cache_v{BOLUS_CACHE_VERSION}_{cfg_hash}"


def save_bolus_result(cache_dir, result):
    """
    Save bolus fitting result to cache.

    Args:
        cache_dir: Cache directory path
        result: Dictionary returned by LogNormalBolusSolver.fit() containing
                'params', 'derived', 'predicted_curve', 'losses',
                'time', 'observed'
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Save NumPy arrays
    np.savez_compressed(
        os.path.join(cache_dir, "arrays.npz"),
        predicted_curve=np.asarray(result['predicted_curve']),
        losses=np.asarray(result['losses']),
        time=np.asarray(result['time']),
        observed=np.asarray(result['observed']),
    )

    # Save scalar parameters as JSON (human-readable)
    scalars = {
        'params': result['params'],
        'derived': result['derived'],
    }
    with open(os.path.join(cache_dir, "params.json"), "w") as f:
        json.dump(scalars, f, indent=2)


def load_bolus_result(cache_dir):
    """
    Load bolus fitting result from cache.

    Args:
        cache_dir: Cache directory path

    Returns:
        Dictionary with the same structure as LogNormalBolusSolver.fit()
        output, or None if the cache file is missing.
    """
    arrays_path = os.path.join(cache_dir, "arrays.npz")
    params_path = os.path.join(cache_dir, "params.json")

    if not os.path.exists(arrays_path) or not os.path.exists(params_path):
        return None

    arrays = np.load(arrays_path)
    with open(params_path, "r") as f:
        scalars = json.load(f)

    return {
        'params': scalars['params'],
        'derived': scalars['derived'],
        'predicted_curve': arrays['predicted_curve'],
        'losses': arrays['losses'].tolist(),
        'time': arrays['time'],
        'observed': arrays['observed'],
    }


def save_bolus_cache_metadata(cache_dir, data_dir, n_svd, roi_coords,
                              smooth_window, fit_config=None):
    """
    Save metadata describing how the cached bolus result was produced.

    Args:
        cache_dir: Cache directory path
        data_dir: Original data directory
        n_svd: SVD components used
        roi_coords: (z_min, z_max, x_min, x_max)
        smooth_window: Smoothing window size
        fit_config: Optional dict of fitting hyper-parameters
    """
    os.makedirs(cache_dir, exist_ok=True)

    metadata = {
        'version': BOLUS_CACHE_VERSION,
        'data_dir': data_dir,
        'n_svd': n_svd,
        'roi_coords': list(roi_coords),
        'smooth_window': smooth_window,
    }
    if fit_config is not None:
        metadata['fit_config'] = fit_config

    with open(os.path.join(cache_dir, "_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def load_bolus_cache_metadata(cache_dir):
    """
    Load bolus cache metadata.

    Returns:
        Metadata dictionary, or None if the file is missing.
    """
    meta_path = os.path.join(cache_dir, "_metadata.json")
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, "r") as f:
        return json.load(f)


def validate_bolus_cache(cache_dir, data_dir, n_svd, roi_coords, smooth_window):
    """
    Check whether an existing bolus cache is still valid.

    Validation checks:
        1. Cache directory exists
        2. Metadata file exists and version matches
        3. Processing parameters match (data_dir, n_svd, ROI, smoothing)
        4. Result files exist

    Args:
        cache_dir: Cache directory path
        data_dir: Current data directory
        n_svd: Current number of SVD components
        roi_coords: Current ROI coordinates tuple
        smooth_window: Current smoothing window

    Returns:
        (is_valid, message) tuple
    """
    if not os.path.exists(cache_dir):
        return False, "Cache directory not found"

    metadata = load_bolus_cache_metadata(cache_dir)
    if metadata is None:
        return False, "Cache metadata not found"

    # Version check
    if metadata.get('version') != BOLUS_CACHE_VERSION:
        return False, (
            f"Cache version mismatch "
            f"(v{metadata.get('version')} vs v{BOLUS_CACHE_VERSION})"
        )

    # Parameter checks
    if metadata.get('data_dir') != data_dir:
        return False, "Data directory mismatch"

    if metadata.get('n_svd') != n_svd:
        return False, (
            f"n_svd mismatch (cached: {metadata.get('n_svd')}, "
            f"requested: {n_svd})"
        )

    if tuple(metadata.get('roi_coords', [])) != tuple(roi_coords):
        return False, "ROI coordinates mismatch"

    if metadata.get('smooth_window') != smooth_window:
        return False, (
            f"smooth_window mismatch "
            f"(cached: {metadata.get('smooth_window')}, "
            f"requested: {smooth_window})"
        )

    # Result files present
    if not os.path.exists(os.path.join(cache_dir, "arrays.npz")):
        return False, "Missing arrays.npz"

    if not os.path.exists(os.path.join(cache_dir, "params.json")):
        return False, "Missing params.json"

    return True, "Valid"


def clear_bolus_cache(cache_dir):
    """Delete bolus cache directory if it exists."""
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"✓ Bolus cache deleted: {cache_dir}")
        return True
    else:
        print(f"No bolus cache to delete: {cache_dir}")
        return False


def get_bolus_cache_size(cache_dir):
    """Get total bolus cache size in MB."""
    if not os.path.exists(cache_dir):
        return 0

    total_size = 0
    for dirpath, _dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)

    return total_size / (1024 * 1024)
