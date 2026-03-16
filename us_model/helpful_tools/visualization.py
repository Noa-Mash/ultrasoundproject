"""
Simple visualization: Compare single files vs concatenated
See if pattern emerges when combining files
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
from scipy.signal import savgol_filter

from data_loader import load_ceus_data
from filters import filter_both


def visualize_all_files_concatenated(data_dir, roi_coords, n_svd=5):
    """
    Simple: Load ALL files, concatenate, show the curve
    """
    
    print("Loading all 107 files...")
    
    # Load ALL files
    all_IQ = []
    for file_num in range(1, 108):
        filename = f"PALA_InVivoRatBrainBolus_{file_num:03d}.mat"
        filepath = os.path.join(data_dir, filename)
        
        if file_num % 10 == 0:
            print(f"  Loading file {file_num}/107...")
        
        data = load_ceus_data(filepath)
        all_IQ.append(data['IQ'])
        params = data['params']
    
    print("\nConcatenating...")
    IQ_full = np.concatenate(all_IQ, axis=2)
    
    total_duration = IQ_full.shape[2] * params['dt_ms'] / 1000.0
    print(f"Total duration: {total_duration:.1f} seconds")
    
    # SVD filter
    print("Applying SVD filtering...")
    tissue, bubbles = filter_both(IQ_full, n_components=n_svd)
    
    # Extract ROI
    z_min, z_max, x_min, x_max = roi_coords
    
    roi_tissue = tissue[z_min:z_max, x_min:x_max, :]
    roi_bubbles = bubbles[z_min:z_max, x_min:x_max, :]
    roi_raw = IQ_full[z_min:z_max, x_min:x_max, :]
    
    raw_curve = np.mean(np.abs(roi_raw), axis=(0, 1))
    tissue_curve = np.mean(np.abs(roi_tissue), axis=(0, 1))
    bubbles_curve = np.mean(np.abs(roi_bubbles), axis=(0, 1))
    
    # Time axis
    time = np.arange(len(tissue_curve)) * params['dt_ms'] / 1000.0
    
    # Smooth
    print("Smoothing...")
    window = 301
    tissue_smooth = savgol_filter(tissue_curve, window_length=window, polyorder=3)
    bubbles_smooth = savgol_filter(bubbles_curve, window_length=window, polyorder=3)
    raw_smooth = savgol_filter(raw_curve, window_length=window, polyorder=3)
    
    # Create visualization
    print("Creating visualization...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    # 1. All three signals - RAW
    ax = axes[0]
    ax.plot(time, raw_curve, alpha=0.3, linewidth=0.5, color='black', label='Raw IQ')
    ax.plot(time, tissue_curve, alpha=0.3, linewidth=0.5, color='blue', label='Tissue')
    ax.plot(time, bubbles_curve, alpha=0.3, linewidth=0.5, color='red', label='Bubbles')
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title('ALL 107 FILES CONCATENATED (~85 seconds) - Raw Signals', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. All three signals - SMOOTHED
    ax = axes[1]
    ax.plot(time, raw_smooth, linewidth=2, color='black', label='Raw IQ (smoothed)', alpha=0.7)
    ax.plot(time, tissue_smooth, linewidth=2, color='blue', label='Tissue (smoothed)', alpha=0.7)
    ax.plot(time, bubbles_smooth, linewidth=2, color='red', label='Bubbles (smoothed)', alpha=0.7)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title('SMOOTHED Signals (looking for bolus pattern)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Mark potential peak
    peak_idx = np.argmax(bubbles_smooth)
    peak_time = time[peak_idx]
    ax.axvline(peak_time, color='orange', linestyle='--', alpha=0.5, label=f'Peak at {peak_time:.1f}s')
    
    # 3. Bubbles only - detailed view
    ax = axes[2]
    ax.plot(time, bubbles_curve, linewidth=0.5, alpha=0.3, color='gray', label='Raw bubbles')
    ax.plot(time, bubbles_smooth, linewidth=3, color='red', label='Smoothed bubbles')
    
    # Stats
    baseline = bubbles_smooth[:50].mean()
    peak = bubbles_smooth.max()
    final = bubbles_smooth[-50:].mean()
    
    ax.axhline(baseline, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label=f'Baseline: {baseline:.1f}')
    ax.axvline(peak_time, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title('BUBBLES Signal - Is this a bolus?', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Analysis box
    enhancement = peak / baseline
    final_peak_ratio = final / peak
    
    is_bolus = (enhancement > 1.5) and (final_peak_ratio < 0.8)
    
    analysis = f'ANALYSIS:\n'
    analysis += f'Duration: {total_duration:.1f}s (107 files)\n'
    analysis += f'Baseline: {baseline:.1f}\n'
    analysis += f'Peak: {peak:.1f} at {peak_time:.1f}s\n'
    analysis += f'Final: {final:.1f}\n'
    analysis += f'Enhancement: {enhancement:.2f}x\n'
    analysis += f'Final/Peak: {final_peak_ratio:.2f}\n\n'
    
    if is_bolus:
        analysis += '✓✓✓ BOLUS PATTERN DETECTED!'
        color = 'lightgreen'
    else:
        analysis += '✗ NO CLEAR BOLUS PATTERN'
        color = 'salmon'
    
    ax.text(0.02, 0.98, analysis, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('all_107_files_concatenated.png', dpi=200, bbox_inches='tight')
    
    print("\n" + "="*70)
    print("RESULT:")
    print("="*70)
    print(f"Enhancement: {enhancement:.2f}x")
    print(f"Final/Peak: {final_peak_ratio:.2f}")
    print(f"Peak at: {peak_time:.1f}s")
    
    if is_bolus:
        print("\n✓✓✓ BOLUS PATTERN DETECTED!")
        print("    This data CAN be used for bolus modeling")
    else:
        print("\n✗ NO CLEAR BOLUS PATTERN")
        print("   This is likely NOT bolus injection data")
    
    print("\n✓ Saved: all_107_files_concatenated.png")
    print("="*70)
    
    plt.show()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    
    DATA_DIR = r"C:\ultrasoundproject\data\PALA_data_InVivoRatBrainBolus_1\IQ"
    ROI_COORDS = (20, 60, 30, 90)
    
    visualize_all_files_concatenated(DATA_DIR, ROI_COORDS, n_svd=5)