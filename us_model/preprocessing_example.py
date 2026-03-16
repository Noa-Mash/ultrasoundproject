"""
Process CEUS data - single file processing with visualization

INSTRUCTIONS:
1. Edit DATA_PATH below to point to your .mat file
2. Run: python process_ceus.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_ceus_data
from filters import filter_both

# Path to single .mat file
DATA_PATH = "data/PALA_data_InVivoRatBrain_1/IQ/PALA_InVivoRatBrain_083.mat"  # ← EDIT THIS PATH

N_COMPONENTS = 10  # ← ADJUST IF NEEDED
FRAME_TO_PLOT = 0  # ← Which frame to visualize

# ============================================================


def plot_results(IQ_complex, tissue_complex, bubbles_complex, frame_idx=0):
    """
    Plot original, tissue, and bubbles side by side.
    Displays magnitude of complex data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    im0 = axes[0].imshow(np.abs(IQ_complex[:, :, frame_idx]), cmap='gray')
    axes[0].set_title(f'Original IQ - Frame {frame_idx}')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0])
    
    # Tissue
    im1 = axes[1].imshow(np.abs(tissue_complex[:, :, frame_idx]), cmap='gray')
    axes[1].set_title(f'Tissue (SVD) - Frame {frame_idx}')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Bubbles
    im2 = axes[2].imshow(np.abs(bubbles_complex[:, :, frame_idx]), cmap='hot')
    axes[2].set_title(f'Microbubbles - Frame {frame_idx}')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.show()


def process_single_file(mat_file: str, n_components: int = 50) -> dict:
    """Process a single .mat file and return structured data."""
    
    print(f"\nProcessing: {Path(mat_file).name}")
    print("-"*60)
    
    # Load complex IQ data
    data = load_ceus_data(mat_file)
    IQ_complex = data['IQ']
    params = data['params']
    
    # Filter using complex data (preserves phase information)
    print(f"Applying SVD filter with {n_components} components...")
    tissue_complex, bubbles_complex = filter_both(IQ_complex, n_components=n_components)
    
    # Build per-frame dictionary
    z, x, t = tissue_complex.shape
    frames = {}
    for frame_idx in range(t):
        frames[frame_idx] = {
            'IQ_complex': IQ_complex[:, :, frame_idx],
            'tissue_complex': tissue_complex[:, :, frame_idx],
            'bubbles_complex': bubbles_complex[:, :, frame_idx]
        }
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"Shape: {tissue_complex.shape}")
    print(f"Frames: {t}")
    print(f"Data type: {tissue_complex.dtype}")
    
    return {
        'IQ_complex': IQ_complex,
        'tissue_complex': tissue_complex,
        'bubbles_complex': bubbles_complex,
        'params': params,
        'frames': frames,
        'filename': Path(mat_file).name
    }


def main():
    print("="*60)
    print("CEUS Data Processing - Single File")
    print("="*60)
    print(f"Input: {DATA_PATH}")
    print(f"SVD components: {N_COMPONENTS}")
    print()
    
    path = Path(DATA_PATH)
    
    # Check if path exists
    if not path.exists():
        print(f"ERROR: Path does not exist: {DATA_PATH}")
        print("Please edit DATA_PATH in this script")
        return None
    
    # Check if it's a .mat file
    if not (path.is_file() and path.suffix == '.mat'):
        print(f"ERROR: Path must be a .mat file")
        print(f"Got: {DATA_PATH}")
        return None
    
    # Process the file
    data = process_single_file(str(path), n_components=N_COMPONENTS)
    
    print()
    print("Access your data:")
    print("  data['IQ_complex']       - original IQ [z,x,t]")
    print("  data['tissue_complex']   - tissue component [z,x,t]")
    print("  data['bubbles_complex']  - microbubbles component [z,x,t]")
    print("  data['frames'][i]        - frame i data")
    print("  data['params']           - acquisition parameters")
    print()
    
    # Visualize
    print(f"Plotting frame {FRAME_TO_PLOT}...")
    plot_results(
        data['IQ_complex'],
        data['tissue_complex'],
        data['bubbles_complex'],
        frame_idx=FRAME_TO_PLOT
    )
    
    return data


if __name__ == '__main__':
    data = main()