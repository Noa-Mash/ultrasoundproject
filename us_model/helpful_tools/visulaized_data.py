"""
Analyze all .mat files to determine if they're sequential or separate injections
Uses your existing data_loader
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import your data loader
US_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if US_MODEL_PATH not in sys.path:
    sys.path.insert(0, US_MODEL_PATH)

from data_loader import load_ceus_data


def analyze_all_files(data_dir, max_files=None, roi=None):
    """
    Analyze baseline and endpoint for all files
    
    Args:
        data_dir: Directory containing .mat files
        max_files: Maximum number of files to analyze (None = all)
        roi: Optional ROI coordinates (z_min, z_max, x_min, x_max)
             If None, uses whole image
    
    Returns:
        results dict with file info
    """
    
    # Get all .mat files
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])
    
    if max_files is not None:
        all_files = all_files[:max_files]
    
    print("="*70)
    print("ANALYZING ALL FILES FOR BASELINE/ENDPOINT PATTERN")
    print("="*70)
    print(f"Directory: {data_dir}")
    print(f"Total files found: {len(all_files)}")
    print(f"Analyzing: {len(all_files)} files")
    
    if roi is not None:
        z_min, z_max, x_min, x_max = roi
        print(f"Using ROI: z=[{z_min}:{z_max}], x=[{x_min}:{x_max}]")
    else:
        print("Using entire image")
    
    print("\nProcessing...")
    
    results = {
        'file_numbers': [],
        'filenames': [],
        'baselines': [],
        'endpoints': [],
        'means': [],
        'stds': [],
        'mins': [],
        'maxs': []
    }
    
    for i, filename in enumerate(all_files):
        filepath = os.path.join(data_dir, filename)
        
        # Progress indicator
        if i % 10 == 0:
            print(f"  Processing file {i+1}/{len(all_files)}... ({filename})")
        
        try:
            # Load data
            data = load_ceus_data(filepath)
            IQ_complex = data['IQ']
            
            # Extract ROI if specified
            if roi is not None:
                z_min, z_max, x_min, x_max = roi
                IQ_roi = IQ_complex[z_min:z_max, x_min:x_max, :]
            else:
                IQ_roi = IQ_complex
            
            # Compute magnitude
            magnitude = np.abs(IQ_roi)
            
            # Baseline: mean of first 50 frames
            baseline = np.mean(magnitude[:, :, :50])
            
            # Endpoint: mean of last 50 frames
            endpoint = np.mean(magnitude[:, :, -50:])
            
            # Overall statistics
            mean_intensity = np.mean(magnitude)
            std_intensity = np.std(magnitude)
            min_intensity = np.min(magnitude)
            max_intensity = np.max(magnitude)
            
            # Extract file number
            try:
                file_num = int(filename.split('_')[-1].replace('.mat', ''))
            except:
                file_num = i
            
            # Store results
            results['file_numbers'].append(file_num)
            results['filenames'].append(filename)
            results['baselines'].append(baseline)
            results['endpoints'].append(endpoint)
            results['means'].append(mean_intensity)
            results['stds'].append(std_intensity)
            results['mins'].append(min_intensity)
            results['maxs'].append(max_intensity)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"\n✓ Successfully processed {len(results['filenames'])} files")
    
    return results


def visualize_results(results):
    """Create visualizations to determine pattern"""
    
    file_nums = np.array(results['file_numbers'])
    baselines = np.array(results['baselines'])
    endpoints = np.array(results['endpoints'])
    means = np.array(results['means'])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Baseline vs File Number
    ax = axes[0, 0]
    ax.plot(file_nums, baselines, 'o-', markersize=5, linewidth=1, label='Baseline (first 50 frames)')
    ax.set_xlabel('File Number', fontsize=12)
    ax.set_ylabel('Baseline Intensity', fontsize=12)
    ax.set_title('Baseline Across Files', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Endpoint vs File Number
    ax = axes[0, 1]
    ax.plot(file_nums, endpoints, 's-', markersize=5, linewidth=1, color='orange', label='Endpoint (last 50 frames)')
    ax.set_xlabel('File Number', fontsize=12)
    ax.set_ylabel('Endpoint Intensity', fontsize=12)
    ax.set_title('Endpoint Across Files', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Baseline vs Endpoint comparison
    ax = axes[1, 0]
    ax.plot(file_nums, baselines, 'o-', markersize=4, linewidth=1, label='Baseline', alpha=0.7)
    ax.plot(file_nums, endpoints, 's-', markersize=4, linewidth=1, label='Endpoint', alpha=0.7)
    ax.set_xlabel('File Number', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title('Baseline vs Endpoint', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Mean intensity across files
    ax = axes[1, 1]
    ax.plot(file_nums, means, 'd-', markersize=5, linewidth=1, color='green', label='Mean intensity')
    ax.set_xlabel('File Number', fontsize=12)
    ax.set_ylabel('Mean Intensity', fontsize=12)
    ax.set_title('Mean Intensity Across Files', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('file_baseline_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved visualization: file_baseline_analysis.png")
    
    return fig


def interpret_results(results):
    """Interpret the pattern and provide conclusion"""
    
    baselines = np.array(results['baselines'])
    endpoints = np.array(results['endpoints'])
    file_nums = np.array(results['file_numbers'])
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Check if files are sequential
    expected_sequence = list(range(file_nums.min(), file_nums.min() + len(file_nums)))
    is_sequential = list(file_nums) == expected_sequence
    
    print(f"\nFile numbering:")
    print(f"  Range: {file_nums.min()} to {file_nums.max()}")
    print(f"  Sequential: {is_sequential}")
    
    # Check baseline trend
    baseline_trend = np.polyfit(file_nums, baselines, 1)[0]  # Slope
    baseline_variation = np.std(baselines) / np.mean(baselines)  # Coefficient of variation
    
    print(f"\nBaseline analysis:")
    print(f"  Mean: {np.mean(baselines):.2f} ± {np.std(baselines):.2f}")
    print(f"  Range: {baselines.min():.2f} - {baselines.max():.2f}")
    print(f"  Trend (slope): {baseline_trend:.4f}")
    print(f"  Variation (CV): {baseline_variation:.2%}")
    
    # Check endpoint trend
    endpoint_trend = np.polyfit(file_nums, endpoints, 1)[0]
    endpoint_variation = np.std(endpoints) / np.mean(endpoints)
    
    print(f"\nEndpoint analysis:")
    print(f"  Mean: {np.mean(endpoints):.2f} ± {np.std(endpoints):.2f}")
    print(f"  Range: {endpoints.min():.2f} - {endpoints.max():.2f}")
    print(f"  Trend (slope): {endpoint_trend:.4f}")
    print(f"  Variation (CV): {endpoint_variation:.2%}")
    
    # Check baseline vs endpoint difference
    baseline_endpoint_diff = endpoints - baselines
    
    print(f"\nBaseline vs Endpoint:")
    print(f"  Mean difference: {np.mean(baseline_endpoint_diff):.2f}")
    print(f"  Files where endpoint > baseline: {np.sum(endpoints > baselines)}/{len(baselines)}")
    
    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if abs(baseline_trend) > 1.0 or abs(endpoint_trend) > 1.0:
        print("\n✓ SEQUENTIAL BOLUS - Files are parts of ONE continuous bolus")
        print("  Evidence:")
        print(f"    - Significant trend in baseline/endpoint across files")
        print(f"    - Baseline slope: {baseline_trend:.2f}")
        print(f"    - Endpoint slope: {endpoint_trend:.2f}")
        print("\n  Recommendation:")
        print("    → CONCATENATE multiple files to see full bolus dynamics")
        print("    → Use files 1-30 for wash-in, 30-60 for peak, 60+ for wash-out")
        
    elif baseline_variation < 0.15 and endpoint_variation < 0.15:
        print("\n✓ SEPARATE INJECTIONS - Each file is independent")
        print("  Evidence:")
        print(f"    - Low variation in baselines ({baseline_variation:.1%})")
        print(f"    - Low variation in endpoints ({endpoint_variation:.1%})")
        print(f"    - No clear trend across files")
        print("\n  Recommendation:")
        print("    → Each file is a separate 0.8s capture")
        print("    → Analyze individually or average multiple files")
        print("    → This might be disruption-replenishment, not bolus")
        
    else:
        print("\n? UNCLEAR PATTERN")
        print("  Evidence:")
        print(f"    - Moderate variation (CV ~{baseline_variation:.1%})")
        print(f"    - Inconsistent trends")
        print("\n  Recommendation:")
        print("    → Examine the plots visually")
        print("    → Check experimental protocol documentation")
        print("    → Look at a few files manually")
    
    print("="*70)


def print_summary_table(results, n_show=20):
    """Print summary table of first N files"""
    
    print("\n" + "="*70)
    print(f"DETAILED RESULTS (First {n_show} files)")
    print("="*70)
    print(f"{'File':<8} {'Baseline':>10} {'Endpoint':>10} {'Diff':>10} {'Mean':>10}")
    print("-"*70)
    
    for i in range(min(n_show, len(results['filenames']))):
        file_num = results['file_numbers'][i]
        baseline = results['baselines'][i]
        endpoint = results['endpoints'][i]
        mean_int = results['means'][i]
        diff = endpoint - baseline
        
        print(f"{file_num:<8} {baseline:>10.2f} {endpoint:>10.2f} {diff:>10.2f} {mean_int:>10.2f}")
    
    if len(results['filenames']) > n_show:
        print(f"... ({len(results['filenames']) - n_show} more files)")
    print("="*70)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == '__main__':
    
    # Configuration
    DATA_DIR = r"C:\ultrasoundproject\data\PALA_data_InVivoRatBrain_1\IQ"
    
    # Optional: Specify ROI (or None for whole image)
    # ROI = (20, 60, 30, 90)  # (z_min, z_max, x_min, x_max)
    ROI = None  # Use whole image
    
    # Optional: Limit number of files to analyze (None = all files)
    MAX_FILES = None  # Analyze all files
    # MAX_FILES = 50  # Or just first 50 files for quick test
    
    # Run analysis
    results = analyze_all_files(DATA_DIR, max_files=MAX_FILES, roi=ROI)
    
    # Show summary table
    print_summary_table(results, n_show=30)
    
    # Visualize
    fig = visualize_results(results)
    
    # Interpret
    interpret_results(results)
    
    # Show plot
    plt.show()
    
    print("\n✓ Analysis complete!")
    print("  Check: file_baseline_analysis.png")