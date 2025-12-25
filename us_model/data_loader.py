"""
Simple data loading for CEUS .mat files
"""

import numpy as np
import scipy.io as sio
from typing import Dict, Tuple


def load_ceus_data(mat_filepath: str) -> Dict:
    """
    Load IQ data and metadata from PALA .mat file.
    
    Args:
        mat_filepath: Path to .mat file
        
    Returns:
        data_dict with:
            - 'IQ_magnitude': Magnitude of IQ [z, x, t], float32
            - 'params': Dictionary with acquisition parameters
    """
    print(f"Loading {mat_filepath}...")
    
    # Load .mat file
    mat_data = sio.loadmat(mat_filepath)
    
    # Extract IQ data
    IQ = mat_data['IQ']  # [z, x, t] complex
    
    # Compute magnitude
    IQ_magnitude = np.abs(IQ).astype(np.float32)
    
    # Extract parameters from UF struct
    UF = mat_data['UF']
    
    params = {
        'TwFreq': float(UF['TwFreq'][0, 0]),           # MHz
        'FrameRateUF': float(UF['FrameRateUF'][0, 0]), # Hz
        'shape': IQ_magnitude.shape,
        'filename': mat_filepath
    }
    
    # Compute derived parameters
    c_sound = 1540  # m/s
    params['wavelength_mm'] = c_sound / (params['TwFreq'] * 1e6) * 1000
    params['dt_ms'] = 1000.0 / params['FrameRateUF']
    
    print(f"  Shape: {params['shape']}")
    print(f"  Frequency: {params['TwFreq']} MHz")
    print(f"  Frame rate: {params['FrameRateUF']} Hz")
    
    return {
        'IQ_magnitude': IQ_magnitude,
        'params': params
    }