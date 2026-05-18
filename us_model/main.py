from cache_utils import get_cache_dir, validate_svd_cache, clear_cache
from ceus_processor import process_and_cache
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from ceus_processor import process_all_files
from select_roi import select_roi_from_data
from LogNormalModel import LogNormalBolusSolver
from LogNormalModel import visualize_fit



# ============================================================================
# MAIN SCRIPT
# ============================================================================
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    
    # CONFIGURATION    
    DATA_DIR = r"C:\ultrasoundproject\data\PALA_data_InVivoRatBrainBolus_1\IQ"
    
    # ROI - Set to None for automatic interactive selection
    ROI_COORDS = None  # Will trigger interactive selection AFTER SVD
    # ROI_COORDS = (100, 300, 150, 350)  # Or specify manually
    
    # Processing
    N_SVD = 5
    SMOOTH_WINDOW = 301
    
    # Fitting
    N_ITERATIONS = 150
    PLOT_EVERY = 10
    PATIENCE = 20
    
    # Cache
    USE_CACHE = True
    FORCE_REPROCESS = False

    # STEP 1: ENSURE SVD CACHE EXISTS
    print("\n" + "="*70)
    print("STEP 1: SVD FILTERING (create cache if needed)")
    print("="*70)
    

    
    all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.mat')])
    cache_dir = get_cache_dir(DATA_DIR, N_SVD)  
    
    # Check if SVD cache exists
    if USE_CACHE and not FORCE_REPROCESS:
        is_valid, msg = validate_svd_cache(cache_dir, DATA_DIR, N_SVD) 
        
        if is_valid:
            print(f"SVD cache exists and is valid")
            print(f"Cache directory: {cache_dir}")
        else:
            print(f"✗ {msg}")
            print(f"\n Running SVD on all {len(all_files)} files...")
            print(f"   This will take 5-10 minutes")
            print(f"   Cache will be ~10-12 GB on disk")
            print(f"   But uses only ~200 MB RAM at a time!\n")
            
            # Run SVD on all files
            params = process_and_cache(DATA_DIR, all_files, N_SVD, cache_dir) #cache_dir
            
            print(f"\n✓ SVD cache created successfully!")
    else:
        if FORCE_REPROCESS:
            print(f" Force reprocessing - running SVD on all files...")
            params = process_and_cache(DATA_DIR, all_files, N_SVD, cache_dir if USE_CACHE else None)
    
    # STEP 2: SELECT ROI (from SVD cache)
    if ROI_COORDS is None:
        print("\n" + "="*70)
        print("STEP 2: INTERACTIVE ROI SELECTION (from SVD cache)")
        print("="*70)
        
        ROI_COORDS = select_roi_from_data(DATA_DIR, n_svd=N_SVD)
        
        print(f"\n✓ ROI selected: {ROI_COORDS}")
    else:
        print("\n" + "="*70)
        print("STEP 2: USING PROVIDED ROI")
        print("="*70)
        print(f"ROI_COORDS: {ROI_COORDS}")

    # STEP 3: EXTRACT ROI AND PROCESS
    print("\n" + "="*70)
    print("STEP 3: EXTRACT ROI + SMOOTH")
    print("="*70)
    
    results = process_all_files(
        data_dir=DATA_DIR,
        n_svd=N_SVD,
        roi_coords=ROI_COORDS,
        smooth_window=SMOOTH_WINDOW,
        use_cache=USE_CACHE,
        force_reprocess=False  # SVD already done!
    )
    
    # STEP 4: FIT LOGNORMAL MODEL
    print("\n" + "="*70)
    print("STEP 4: FIT LOGNORMAL MODEL")
    print("="*70)
    
    solver = LogNormalBolusSolver()
    
    fit_results = solver.fit(
        results['time'],
        results['bubbles_smooth'],
        n_iterations=N_ITERATIONS,
        lr=0.01,
        plot_every=PLOT_EVERY,
        patience=PATIENCE,
        verbose=True
    )
    
    # STEP 5: VISUALIZE
    print("\n" + "="*70)
    print("STEP 5: VISUALIZE")
    print("="*70)
    
    fig = visualize_fit(fit_results, title=f"Lognormal Fit - {results['n_files']} files")
    plt.savefig('lognormal_fit_final.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: lognormal_fit_final.png")
    
    plt.show()
    
    # SUMMARY
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"ROI used: {results['roi_coords']}")
    print(f"Files processed: {results['n_files']}")
    print(f"Duration: {results['time'][-1]:.1f}s")
    print(f"Enhancement: {results['bubbles_smooth'].max() / results['bubbles_smooth'][:50].mean():.2f}x")
    print(f"Peak time: {fit_results['derived']['t_peak']:.2f}s")
    print(f"R²: {1 - np.var(fit_results['observed'] - fit_results['predicted_curve'])/np.var(fit_results['observed']):.4f}")
    print("="*70)