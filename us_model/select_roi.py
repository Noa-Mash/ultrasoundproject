"""
Interactive ROI Selector for CEUS Data
Loads SVD cache (if exists) or first file for ROI selection
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from data_loader import load_ceus_data
from filters import filter_both
from cache_utils import get_cache_dir, load_svd_file, validate_svd_cache, load_cache_metadata

class InteractiveROISelector:
    """Interactive ROI selection from image"""
    
    def __init__(self, image, title="Select ROI"):
        self.image = image
        self.roi_coords = None
        
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.im = self.ax.imshow(image, cmap='hot', aspect='auto')
        self.ax.set_title('CLICK AND DRAG to select ROI, then CLOSE window', 
                         fontsize=14, fontweight='bold', color='red')
        self.ax.set_xlabel('Lateral (x)')
        self.ax.set_ylabel('Axial (z)')
        plt.colorbar(self.im, ax=self.ax)
        
        instructions = """
        SELECT ROI
        
        1. Click and drag to draw rectangle
        2. Redraw if needed
        3. Close window when satisfied
        """
        self.ax.text(0.02, 0.98, instructions,
                    transform=self.ax.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        self.selector = RectangleSelector(
            self.ax, self.on_select,
            useblit=True, button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels', interactive=True,
            props=dict(facecolor='red', edgecolor='red', alpha=0.3, fill=True)
        )
        
    def on_select(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        x_min, x_max = min(x1, x2), max(x1, x2)
        z_min, z_max = min(y1, y2), max(y1, y2)
        
        self.roi_coords = (z_min, z_max, x_min, x_max)
        
        self.ax.set_title(
            f'ROI: z=[{z_min}:{z_max}], x=[{x_min}:{x_max}] | Close to confirm',
            fontsize=12, fontweight='bold', color='green'
        )
        self.fig.canvas.draw()
        
        print(f"\n  ✓ Selected ROI: z=[{z_min}:{z_max}], x=[{x_min}:{x_max}]")
    
    def get_roi(self):
        plt.show()
        return self.roi_coords


def select_roi_from_data(data_dir, n_svd=5):
    """
    Select ROI - tries to load from SVD cache first, falls back to first file
    """
    
    print("="*70)
    print("ROI SELECTOR")
    print("="*70)
    
    # Find files
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])
    if len(all_files) == 0:
        raise ValueError("No .mat files found in directory!")
    
    print(f"\nData directory: {data_dir}")
    print(f"Total files: {len(all_files)}")
    
    # Try to load from SVD cache first
    cache_dir = get_cache_dir(data_dir, n_svd)
    is_valid, msg = validate_svd_cache(cache_dir, data_dir, n_svd)
    
    if is_valid:
        print(f"\n✓ Found SVD cache - loading first file for ROI selection...")
        
        # Load first file from cache
        bubbles, tissue = load_svd_file(cache_dir, all_files[0])
        
        if bubbles is not None:
            bubbles_avg = np.mean(np.abs(bubbles), axis=2)
            source = "SVD cache"
        else:
            raise ValueError("Failed to load from cache")
    else:
        print(f"\n✗ No SVD cache found - loading first file...")
        first_file = all_files[0]
        print(f"  File: {first_file}")
        
        filepath = os.path.join(data_dir, first_file)
        data = load_ceus_data(filepath)
        IQ = data['IQ']
        
        print(f"  Applying SVD (n_svd={n_svd})...")
        tissue, bubbles = filter_both(IQ, n_components=n_svd)
        
        bubbles_avg = np.mean(np.abs(bubbles), axis=2)
        source = "first file"
    
    print(f"\n  Image shape: {bubbles_avg.shape}")
    print(f"  Intensity range: [{bubbles_avg.min():.1f}, {bubbles_avg.max():.1f}]")
    print(f"  Source: {source}")
    
    # Interactive ROI selection
    print(f"\nOpening ROI selector window...")
    
    selector = InteractiveROISelector(bubbles_avg, title=f"Select ROI ({source})")
    roi_coords = selector.get_roi()
    
    if roi_coords is None:
        print("\n⚠️  No ROI selected, using default central region")
        z, x = bubbles_avg.shape
        roi_coords = (z//3, 2*z//3, x//3, 2*x//3)
    
    z_min, z_max, x_min, x_max = roi_coords
    
    print("\n" + "="*70)
    print("ROI SELECTION COMPLETE!")
    print("="*70)
    print(f"\nSelected ROI coordinates:")
    print(f"  ROI_COORDS = ({z_min}, {z_max}, {x_min}, {x_max})")
    print(f"\nROI size: {z_max-z_min} × {x_max-x_min} pixels")
    print("="*70)
    
    return roi_coords


# Backward compatibility
select_roi_from_cache = select_roi_from_data
select_roi_from_first_file = select_roi_from_data
select_roi_from_raw_data = select_roi_from_data


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    
    DATA_DIR = r"C:\ultrasoundproject\data\PALA_data_InVivoRatBrainBolus_1\IQ"
    N_SVD = 5
    
    roi_coords = select_roi_from_data(DATA_DIR, n_svd=N_SVD)
    
    print("\n✓ Done! Use ROI_COORDS in your LogNormalModel.py")