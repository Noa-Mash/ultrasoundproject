"""
PALA-style clutter filtering using SVD
Separates tissue (coherent, slow) from microbubbles (sparse, fast)
CORRECT VERSION: Works with complex IQ data to preserve phase information
"""

import numpy as np


def filter_tissue(IQ_complex: np.ndarray, n_components: int = 50) -> np.ndarray:
    """
    Extract tissue component using SVD on complex IQ data.
    Tissue = coherent, slow-moving signal captured by first N singular values.
    
    Args:
        IQ_complex: Complex IQ data [z, x, t] (complex128 or complex64)
        n_components: Number of SVD components for tissue (default: 50)
        
    Returns:
        tissue: Tissue component [z, x, t], complex64
    """
    print(f"Filtering tissue (SVD with {n_components} components)...")
    
    z, x, t = IQ_complex.shape
    
    # Reshape to matrix: [pixels, time]
    M = IQ_complex.reshape(-1, t)
    
    # SVD decomposition on complex data
    print("  Computing SVD...")
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    # Keep only first n_components for tissue
    n_keep = min(n_components, len(S))
    S_tissue = np.zeros_like(S)
    S_tissue[:n_keep] = S[:n_keep]
    
    # Reconstruct tissue (preserves complex values)
    M_tissue = U @ np.diag(S_tissue) @ Vt
    
    # Reshape back to [z, x, t]
    tissue = M_tissue.reshape(z, x, t).astype(np.complex64)
    
    # Log energy captured
    energy_ratio = np.sum(S_tissue**2) / np.sum(S**2) * 100
    print(f"  Tissue captures {energy_ratio:.1f}% of signal energy")
    
    return tissue


def filter_microbubbles(IQ_complex: np.ndarray, n_components: int = 50) -> np.ndarray:
    """
    Extract microbubble component using SVD on complex IQ data.
    Bubbles = fast, sparse signal in remaining SVD components.
    
    Args:
        IQ_complex: Complex IQ data [z, x, t] (complex128 or complex64)
        n_components: Number of SVD components to remove as tissue (default: 50)
        
    Returns:
        bubbles: Microbubble component [z, x, t], complex64
    """
    print(f"Filtering microbubbles (removing {n_components} tissue components)...")
    
    z, x, t = IQ_complex.shape
    
    # Reshape to matrix: [pixels, time]
    M = IQ_complex.reshape(-1, t)
    
    # SVD decomposition on complex data
    print("  Computing SVD...")
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    # Remove first n_components (tissue)
    n_remove = min(n_components, len(S))
    S_bubbles = S.copy()
    S_bubbles[:n_remove] = 0
    
    # Reconstruct bubbles (preserves complex values)
    M_bubbles = U @ np.diag(S_bubbles) @ Vt
    
    # Reshape back to [z, x, t]
    bubbles = M_bubbles.reshape(z, x, t).astype(np.complex64)
    
    # Log energy captured
    energy_ratio = np.sum(S_bubbles**2) / np.sum(S**2) * 100
    print(f"  Bubbles capture {energy_ratio:.1f}% of signal energy")
    
    return bubbles


def filter_both(IQ_complex: np.ndarray, n_components: int = 50) -> tuple:
    """
    Extract both tissue and microbubbles in one pass (more efficient).
    
    Args:
        IQ_complex: Complex IQ data [z, x, t] (complex128 or complex64)
        n_components: Number of SVD components for tissue
        
    Returns:
        tissue: Tissue component [z, x, t], complex64
        bubbles: Microbubble component [z, x, t], complex64
    """
    print(f"Filtering tissue and microbubbles (SVD with {n_components} components)...")
    
    z, x, t = IQ_complex.shape
    
    # Reshape to matrix
    M = IQ_complex.reshape(-1, t)
    
    # SVD decomposition on complex data
    print("  Computing SVD...")
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    n_keep = min(n_components, len(S))
    
    # Tissue: first n_keep components
    S_tissue = np.zeros_like(S)
    S_tissue[:n_keep] = S[:n_keep]
    M_tissue = U @ np.diag(S_tissue) @ Vt
    tissue = M_tissue.reshape(z, x, t).astype(np.complex64)
    
    # Bubbles: remaining components (preserves complex values)
    M_bubbles = M - M_tissue
    bubbles = M_bubbles.reshape(z, x, t).astype(np.complex64)
    
    # Log results
    tissue_energy = np.sum(S_tissue**2) / np.sum(S**2) * 100
    bubble_energy = 100 - tissue_energy
    print(f"  Tissue: {tissue_energy:.1f}% energy")
    print(f"  Bubbles: {bubble_energy:.1f}% energy")
    
    return tissue, bubbles