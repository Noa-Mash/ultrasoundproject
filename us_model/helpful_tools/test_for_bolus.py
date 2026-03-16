"""
Comprehensive CEUS Bolus Diagnostics
Tests different filtering and smoothing strategies
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


def comprehensive_bolus_diagnostic(mat_file, roi_coords, 
                                   svd_components_list=[3, 5, 10, 15, 20],
                                   smooth_windows=[51, 101, 151, 201]):
    """
    Test different combinations of SVD filtering and temporal smoothing
    to find the best way to reveal bolus dynamics
    
    Args:
        mat_file: Path to .mat file
        roi_coords: (z_min, z_max, x_min, x_max)
        svd_components_list: List of n_components to test
        smooth_windows: List of smoothing windows to test
    """
    
    print("="*70)
    print("COMPREHENSIVE BOLUS DIAGNOSTIC")
    print("="*70)
    
    # Load data
    print("\n[1] Loading data...")
    data = load_ceus_data(mat_file)
    IQ_complex = data['IQ']
    params = data['params']
    
    z_min, z_max, x_min, x_max = roi_coords
    
    # Extract time axis
    n_frames = IQ_complex.shape[2]
    time_axis = np.arange(n_frames) * params['dt_ms'] / 1000.0
    
    # ================================================================
    # PART 1: Test different SVD filtering strategies
    # ================================================================
    print("\n[2] Testing SVD filtering strategies...")
    
    fig1, axes = plt.subplots(len(svd_components_list) + 1, 1, 
                             figsize=(14, 3*(len(svd_components_list)+1)))
    
    # First: Raw IQ (no filtering)
    print("  - Raw IQ (no SVD)")
    roi_raw = IQ_complex[z_min:z_max, x_min:x_max, :]
    raw_curve = np.mean(np.abs(roi_raw), axis=(0, 1))
    
    ax = axes[0]
    ax.plot(time_axis, raw_curve, linewidth=1, alpha=0.7, color='black')
    ax.set_ylabel('Intensity')
    ax.set_title('RAW IQ (no filtering)', fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    baseline_raw = raw_curve[:50].mean()
    peak_raw = raw_curve.max()
    final_raw = raw_curve[-50:].mean()
    ax.text(0.02, 0.95, f'Enhancement: {peak_raw/baseline_raw:.2f}x\nFinal/Peak: {final_raw/peak_raw:.2f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Then: Different SVD components
    curves_dict = {'raw': raw_curve}
    
    for i, n_comp in enumerate(svd_components_list):
        print(f"  - SVD with n={n_comp} components")
        
        tissue, bubbles = filter_both(IQ_complex, n_components=n_comp)
        
        # Extract BOTH tissue and bubbles curves
        roi_tissue = tissue[z_min:z_max, x_min:x_max, :]
        roi_bubbles = bubbles[z_min:z_max, x_min:x_max, :]
        
        tissue_curve = np.mean(np.abs(roi_tissue), axis=(0, 1))
        bubbles_curve = np.mean(np.abs(roi_bubbles), axis=(0, 1))
        
        curves_dict[f'tissue_n{n_comp}'] = tissue_curve
        curves_dict[f'bubbles_n{n_comp}'] = bubbles_curve
        
        # Plot BUBBLES
        ax = axes[i+1]
        ax.plot(time_axis, bubbles_curve, linewidth=1, alpha=0.7, color='red', label='Bubbles')
        ax.plot(time_axis, tissue_curve, linewidth=1, alpha=0.4, color='blue', label='Tissue')
        ax.set_ylabel('Intensity')
        ax.set_title(f'SVD n={n_comp} (Bubbles vs Tissue)', fontweight='bold', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Statistics for BUBBLES
        baseline_b = bubbles_curve[:50].mean()
        peak_b = bubbles_curve.max()
        final_b = bubbles_curve[-50:].mean()
        
        # Statistics for TISSUE
        baseline_t = tissue_curve[:50].mean()
        peak_t = tissue_curve.max()
        final_t = tissue_curve[-50:].mean()
        
        stats_text = f'BUBBLES: {peak_b/baseline_b:.2f}x, F/P={final_b/peak_b:.2f}\n'
        stats_text += f'TISSUE: {peak_t/baseline_t:.2f}x, F/P={final_t/peak_t:.2f}'
        
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    plt.tight_layout()
    plt.savefig('svd_comparison.png', dpi=200)
    print("  ✓ Saved: svd_comparison.png")
    
    # ================================================================
    # PART 2: Test temporal smoothing on best candidate
    # ================================================================
    print("\n[3] Testing temporal smoothing...")
    
    # Find which signal shows best enhancement
    best_signal = 'raw'
    best_enhancement = peak_raw / baseline_raw
    
    for key, curve in curves_dict.items():
        if 'bubbles' in key or 'tissue' in key:
            baseline = curve[:50].mean()
            peak = curve.max()
            enhancement = peak / baseline
            
            if enhancement > best_enhancement:
                best_signal = key
                best_enhancement = enhancement
    
    print(f"  Best signal: {best_signal} (enhancement: {best_enhancement:.2f}x)")
    best_curve = curves_dict[best_signal]
    
    # Test different smoothing windows
    fig2, axes = plt.subplots(len(smooth_windows), 1, 
                             figsize=(14, 3*len(smooth_windows)))
    
    for i, window in enumerate(smooth_windows):
        if window >= len(best_curve):
            window = len(best_curve) - 1
            if window % 2 == 0:
                window -= 1
        
        # Apply smoothing
        smoothed = savgol_filter(best_curve, window_length=window, polyorder=3)
        
        ax = axes[i]
        ax.plot(time_axis, best_curve, linewidth=0.5, alpha=0.3, 
                color='gray', label='Raw')
        ax.plot(time_axis, smoothed, linewidth=2, color='red', 
                label=f'Smoothed (w={window})')
        
        ax.set_ylabel('Intensity')
        ax.set_title(f'Smoothing window = {window} frames ({window}ms)', 
                    fontweight='bold', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Check if bolus pattern emerges
        baseline_s = smoothed[:50].mean()
        peak_s = smoothed.max()
        final_s = smoothed[-50:].mean()
        peak_idx = np.argmax(smoothed)
        peak_time = time_axis[peak_idx]
        
        # Is there a clear rise and fall?
        is_bolus = (final_s / peak_s < 0.8) and (peak_time > 0.1) and (peak_time < 0.7)
        
        stats_text = f'Peak: {peak_s:.1f} at {peak_time:.2f}s\n'
        stats_text += f'Enhancement: {peak_s/baseline_s:.2f}x\n'
        stats_text += f'Final/Peak: {final_s/peak_s:.2f}\n'
        stats_text += f'Pattern: {"BOLUS ✓" if is_bolus else "UNCLEAR"}'
        
        color = 'lightgreen' if is_bolus else 'wheat'
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
    
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    plt.tight_layout()
    plt.savefig('smoothing_comparison.png', dpi=200)
    print("  ✓ Saved: smoothing_comparison.png")
    
    # ================================================================
    # PART 3: Recommendation
    # ================================================================
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    # Test all combinations
    best_combo = None
    best_score = 0
    
    for signal_key, curve in curves_dict.items():
        baseline = curve[:50].mean()
        peak = curve.max()
        final = curve[-50:].mean()
        peak_idx = np.argmax(curve)
        
        # Score based on:
        # 1. Enhancement ratio
        # 2. Clear peak position (not at edges)
        # 3. Final < Peak (wash-out)
        
        enhancement = peak / baseline
        peak_position_score = 1.0 if (0.2 < time_axis[peak_idx] < 0.6) else 0.5
        washout_score = max(0, (1 - final/peak)) * 2  # Higher if clear wash-out
        
        score = enhancement * peak_position_score * (1 + washout_score)
        
        if score > best_score:
            best_score = score
            best_combo = signal_key
    
    print(f"\nBest signal source: {best_combo}")
    print(f"Score: {best_score:.2f}")
    
    # Get the best curve
    best_curve_final = curves_dict[best_combo]
    
    # Find best smoothing
    best_window = None
    best_bolus_score = 0
    
    for window in smooth_windows:
        if window >= len(best_curve_final):
            continue
        if window % 2 == 0:
            window -= 1
        
        smoothed = savgol_filter(best_curve_final, window_length=window, polyorder=3)
        
        baseline = smoothed[:50].mean()
        peak = smoothed.max()
        final = smoothed[-50:].mean()
        peak_idx = np.argmax(smoothed)
        peak_time = time_axis[peak_idx]
        
        # Bolus score
        if 0.1 < peak_time < 0.7 and final/peak < 0.9:
            bolus_score = (peak/baseline) * (1 - final/peak) * 10
            
            if bolus_score > best_bolus_score:
                best_bolus_score = bolus_score
                best_window = window
    
    print(f"Best smoothing window: {best_window} frames")
    
    print("\n" + "-"*70)
    print("RECOMMENDED WORKFLOW:")
    print("-"*70)
    
    if 'raw' in best_combo:
        print("1. Use RAW IQ (skip SVD filtering)")
    elif 'tissue' in best_combo:
        n = int(best_combo.split('_n')[1])
        print(f"1. Use SVD TISSUE component (n={n})")
        print("   → Bolus is slow/coherent signal, captured in tissue!")
    else:
        n = int(best_combo.split('_n')[1])
        print(f"1. Use SVD BUBBLES component (n={n})")
    
    if best_window:
        print(f"2. Apply Savitzky-Golay smoothing (window={best_window})")
        print(f"3. Fit lognormal bolus model to SMOOTHED curve")
    else:
        print("2. Data might not show clear bolus pattern")
        print("   → Try larger smoothing windows (201, 301)")
        print("   → Or concatenate multiple files")
    
    print("="*70)
    
    plt.show()
    
    return {
        'curves': curves_dict,
        'best_signal': best_combo,
        'best_window': best_window,
        'time': time_axis
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    
    MAT_FILE = r"C:\ultrasoundproject\data\PALA_data_InVivoRatBrainBolus_1\IQ\PALA_InVivoRatBrainBolus_105.mat"
    
    # Use your ROI
    ROI_COORDS = (20, 60, 30, 90)
    
    # Run comprehensive diagnostic
    results = comprehensive_bolus_diagnostic(
        mat_file=MAT_FILE,
        roi_coords=ROI_COORDS,
        svd_components_list=[3, 5, 10, 15, 20],  # Test different filtering
        smooth_windows=[51, 101, 151, 201]        # Test different smoothing
    )
    
    print("\n✓ Analysis complete!")
    print("  Check outputs:")
    print("    - svd_comparison.png (which filter works best)")
    print("    - smoothing_comparison.png (which smoothing reveals bolus)")