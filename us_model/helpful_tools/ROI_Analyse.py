"""
Load all files → SVD each separately → Extract ROI → Concatenate curves
Interactive ROI selection ONCE at the start
"""
import sys
import os

US_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

if US_MODEL_PATH not in sys.path:
    sys.path.insert(0, US_MODEL_PATH)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from scipy.signal import savgol_filter

from data_loader import load_ceus_data
from filters import filter_both


class InteractiveROISelector:
    """Interactive ROI selection"""
    
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
        SELECT ROI ONCE (applies to all files)
        
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
        
        print(f"  Selected ROI: z=[{z_min}:{z_max}], x=[{x_min}:{x_max}]")
    
    def get_roi(self):
        plt.show()
        return self.roi_coords


def process_all_files_with_roi(data_dir, n_svd=5, smooth_window=301, 
                                roi_coords=None, interactive_roi=True):
    """
    Process all .mat files in directory
    
    Args:
        data_dir: Directory containing .mat files
        n_svd: Number of SVD components for tissue
        smooth_window: Smoothing window size
        roi_coords: (z_min, z_max, x_min, x_max) or None for interactive
        interactive_roi: If True, use interactive ROI selection
        
    Returns:
        dict with time, curves, and metadata
    """
    
    print("="*70)
    print("PROCESSING ALL FILES WITH SINGLE ROI")
    print("="*70)
    
    # Find all .mat files
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])
    n_files = len(all_files)
    
    print(f"\nFound {n_files} .mat files")
    print(f"  First: {all_files[0]}")
    print(f"  Last: {all_files[-1]}")
    
    if n_files == 0:
        raise ValueError("No .mat files found!")
    
    # ================================================================
    # STEP 1: Interactive ROI selection (if needed)
    # ================================================================
    
    if roi_coords is None and interactive_roi:
        print("\n[1] Interactive ROI selection...")
        print("    Loading first file for ROI selection...")
        
        # Load first file
        first_file = os.path.join(data_dir, all_files[0])
        data = load_ceus_data(first_file)
        IQ = data['IQ']
        params = data['params']
        
        # Apply SVD to first file to show bubbles
        print("    Applying SVD to show bubbles...")
        tissue, bubbles = filter_both(IQ, n_components=n_svd)
        
        # Show time-averaged bubbles for ROI selection
        bubbles_avg = np.mean(np.abs(bubbles), axis=2)
        
        print("    Opening ROI selector...")
        selector = InteractiveROISelector(bubbles_avg, title="Select ROI (applies to ALL files)")
        roi_coords = selector.get_roi()
        
        if roi_coords is None:
            print("\n  No ROI selected, using default central region")
            z, x, t = IQ.shape
            z_min, z_max = z//3, 2*z//3
            x_min, x_max = x//3, 2*x//3
            roi_coords = (z_min, z_max, x_min, x_max)
    
    elif roi_coords is None:
        # Default ROI
        first_file = os.path.join(data_dir, all_files[0])
        data = load_ceus_data(first_file)
        IQ = data['IQ']
        params = data['params']
        
        z, x, t = IQ.shape
        z_min, z_max = z//3, 2*z//3
        x_min, x_max = x//3, 2*x//3
        roi_coords = (z_min, z_max, x_min, x_max)
        print(f"\n[1] Using default ROI: {roi_coords}")
    else:
        print(f"\n[1] Using provided ROI: {roi_coords}")
    
    z_min, z_max, x_min, x_max = roi_coords
    print(f"    ROI: z=[{z_min}:{z_max}], x=[{x_min}:{x_max}]")
    print(f"    Size: {z_max-z_min} × {x_max-x_min} pixels")
    
    # ================================================================
    # STEP 2: Process each file separately
    # ================================================================
    
    print(f"\n[2] Processing {n_files} files...")
    print("    Each file: Load → SVD → Extract ROI")
    
    all_tissue_curves = []
    all_bubbles_curves = []
    all_raw_curves = []
    
    for i, filename in enumerate(all_files):
        if i % 10 == 0 or i == n_files - 1:
            print(f"    Processing file {i+1}/{n_files}: {filename}")
        
        filepath = os.path.join(data_dir, filename)
        
        try:
            # Load
            data = load_ceus_data(filepath)
            IQ = data['IQ']
            params = data['params']
            
            # SVD filter
            tissue, bubbles = filter_both(IQ, n_components=n_svd)
            
            # Extract ROI curves
            roi_raw = IQ[z_min:z_max, x_min:x_max, :]
            roi_tissue = tissue[z_min:z_max, x_min:x_max, :]
            roi_bubbles = bubbles[z_min:z_max, x_min:x_max, :]
            
            raw_curve = np.mean(np.abs(roi_raw), axis=(0, 1))
            tissue_curve = np.mean(np.abs(roi_tissue), axis=(0, 1))
            bubbles_curve = np.mean(np.abs(roi_bubbles), axis=(0, 1))
            
            all_raw_curves.append(raw_curve)
            all_tissue_curves.append(tissue_curve)
            all_bubbles_curves.append(bubbles_curve)
            
        except Exception as e:
            print(f"    ⚠️  Error processing {filename}: {e}")
            continue
    
    print(f"    ✓ Successfully processed {len(all_bubbles_curves)} files")
    
    # ================================================================
    # STEP 3: Concatenate curves
    # ================================================================
    
    print("\n[3] Concatenating curves...")
    
    raw_full = np.concatenate(all_raw_curves)
    tissue_full = np.concatenate(all_tissue_curves)
    bubbles_full = np.concatenate(all_bubbles_curves)
    
    # Time axis
    time_full = np.arange(len(bubbles_full)) * params['dt_ms'] / 1000.0
    duration = time_full[-1]
    
    print(f"    Total frames: {len(bubbles_full)}")
    print(f"    Total duration: {duration:.1f} seconds")
    
    # ================================================================
    # STEP 4: Apply smoothing
    # ================================================================
    
    print(f"\n[4] Applying smoothing (window={smooth_window})...")
    
    if smooth_window >= len(bubbles_full):
        smooth_window = len(bubbles_full) - 1
        if smooth_window % 2 == 0:
            smooth_window -= 1
        print(f"    Adjusted window to {smooth_window}")
    
    raw_smooth = savgol_filter(raw_full, window_length=smooth_window, polyorder=3)
    tissue_smooth = savgol_filter(tissue_full, window_length=smooth_window, polyorder=3)
    bubbles_smooth = savgol_filter(bubbles_full, window_length=smooth_window, polyorder=3)
    
    # ================================================================
    # STEP 5: Analyze
    # ================================================================
    
    print("\n[5] Analyzing signal...")
    
    baseline = bubbles_smooth[:50].mean()
    peak = bubbles_smooth.max()
    final = bubbles_smooth[-50:].mean()
    peak_idx = np.argmax(bubbles_smooth)
    peak_time = time_full[peak_idx]
    
    enhancement = peak / baseline
    final_peak_ratio = final / peak
    
    print(f"\n    Baseline: {baseline:.1f}")
    print(f"    Peak: {peak:.1f} at {peak_time:.1f}s")
    print(f"    Final: {final:.1f}")
    print(f"    Enhancement: {enhancement:.2f}x")
    print(f"    Final/Peak: {final_peak_ratio:.2f}")
    
    is_bolus = (enhancement > 1.5) and (final_peak_ratio < 0.8) and (0.1*duration < peak_time < 0.8*duration)
    
    if is_bolus:
        print(f"\n    ✓✓✓ BOLUS PATTERN DETECTED!")
    else:
        print(f"\n    ✗ No clear bolus pattern")
    
    print("="*70)
    
    return {
        'time': time_full,
        'raw': raw_full,
        'tissue': tissue_full,
        'bubbles': bubbles_full,
        'raw_smooth': raw_smooth,
        'tissue_smooth': tissue_smooth,
        'bubbles_smooth': bubbles_smooth,
        'roi_coords': roi_coords,
        'n_files': len(all_bubbles_curves),
        'params': params,
        'is_bolus': is_bolus,
        'enhancement': enhancement,
        'peak_time': peak_time,
        'final_peak_ratio': final_peak_ratio
    }


def visualize_results(results, save_path='all_files_processed.png'):
    """Visualize the processed data"""
    
    time = results['time']
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # 1. All signals - RAW
    ax = axes[0]
    ax.plot(time, results['raw'], alpha=0.3, linewidth=0.5, color='black', label='Raw IQ')
    ax.plot(time, results['tissue'], alpha=0.3, linewidth=0.5, color='blue', label='Tissue')
    ax.plot(time, results['bubbles'], alpha=0.3, linewidth=0.5, color='red', label='Bubbles')
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title(f'ALL {results["n_files"]} FILES PROCESSED - Raw Signals', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. All signals - SMOOTHED
    ax = axes[1]
    ax.plot(time, results['raw_smooth'], linewidth=2, color='black', label='Raw IQ', alpha=0.7)
    ax.plot(time, results['tissue_smooth'], linewidth=2, color='blue', label='Tissue', alpha=0.7)
    ax.plot(time, results['bubbles_smooth'], linewidth=2, color='red', label='Bubbles', alpha=0.7)
    
    ax.axvline(results['peak_time'], color='orange', linestyle='--', alpha=0.5, linewidth=2)
    
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title('SMOOTHED Signals', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 3. Bubbles detail with analysis
    ax = axes[2]
    ax.plot(time, results['bubbles'], linewidth=0.5, alpha=0.2, color='gray', label='Raw')
    ax.plot(time, results['bubbles_smooth'], linewidth=3, color='red', label='Smoothed')
    
    baseline = results['bubbles_smooth'][:50].mean()
    ax.axhline(baseline, color='green', linestyle='--', alpha=0.5, linewidth=2, label=f'Baseline: {baseline:.1f}')
    ax.axvline(results['peak_time'], color='orange', linestyle='--', alpha=0.5, linewidth=2, 
              label=f'Peak: {results["peak_time"]:.1f}s')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title('BUBBLES Signal - Analysis', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Analysis box
    analysis = f'ANALYSIS:\n'
    analysis += f'Files processed: {results["n_files"]}\n'
    analysis += f'Duration: {time[-1]:.1f}s\n'
    analysis += f'ROI: z=[{results["roi_coords"][0]}:{results["roi_coords"][1]}], '
    analysis += f'x=[{results["roi_coords"][2]}:{results["roi_coords"][3]}]\n\n'
    analysis += f'Peak: {results["bubbles_smooth"].max():.1f} at {results["peak_time"]:.1f}s\n'
    analysis += f'Enhancement: {results["enhancement"]:.2f}x\n'
    analysis += f'Final/Peak: {results["final_peak_ratio"]:.2f}\n\n'
    
    if results['is_bolus']:
        analysis += '✓✓✓ BOLUS PATTERN!'
        color = 'lightgreen'
    else:
        analysis += '✗ NO BOLUS PATTERN'
        color = 'salmon'
    
    ax.text(0.02, 0.98, analysis, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    
    plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    
    # Configuration
    DATA_DIR = r"C:\ultrasoundproject\data\PALA_data_InVivoRatBrainBolus_1\IQ"
    
    # Process all files with interactive ROI
    results = process_all_files_with_roi(
        data_dir=DATA_DIR,
        n_svd=5,                    # SVD components
        smooth_window=301,          # Smoothing window
        roi_coords=None,            # None = interactive selection
        interactive_roi=True        # Enable interactive ROI
    )
    
    # Visualize
    visualize_results(results, save_path='all_files_processed.png')
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Processed {results['n_files']} files")
    print(f"Total duration: {results['time'][-1]:.1f} seconds")
    print(f"Enhancement: {results['enhancement']:.2f}x")
    print(f"Bolus pattern: {'YES ✓' if results['is_bolus'] else 'NO ✗'}")
    print("="*70)