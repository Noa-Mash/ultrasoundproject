"""
CEUS Bolus Pipeline
Standalone script that orchestrates the full processing pipeline:
  1. SVD filtering (cached via cache_utils)
  2. ROI selection (interactive or manual)
  3. ROI extraction + temporal smoothing
  4. Lognormal bolus model fitting (cached via bolus_cache)
  5. Visualization and summary

Run directly:  python run_bolus_pipeline.py
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from cache_utils import get_cache_dir, validate_svd_cache
from ceus_processor import process_and_cache, process_all_files
from select_roi import select_roi_from_data
from LogNormalModel import LogNormalBolusSolver, visualize_fit
from bolus_cache import (
    get_bolus_cache_dir,
    validate_bolus_cache,
    save_bolus_result,
    load_bolus_result,
    save_bolus_cache_metadata,
    get_bolus_cache_size,
)


def run_pipeline(data_dir, roi_coords=None, n_svd=5, smooth_window=301,
                 n_iterations=150, lr=0.01, plot_every=10, patience=20,
                 use_cache=True, force_reprocess=False):
    """
    Run the full CEUS bolus processing and fitting pipeline.

    Args:
        data_dir: Directory containing .mat IQ files
        roi_coords: (z_min, z_max, x_min, x_max) or None for interactive
        n_svd: Number of SVD components for tissue/bubble separation
        smooth_window: Savitzky-Golay smoothing window size
        n_iterations: Max fitting iterations
        lr: Learning rate for the optimizer
        plot_every: Plot every N iterations (0 to disable)
        patience: Early stopping patience
        use_cache: Use cached SVD / bolus results when available
        force_reprocess: Force re-running SVD even if cache exists

    Returns:
        (results, fit_results) tuple
            results     – dict from process_all_files (time-series, ROI, etc.)
            fit_results – dict from LogNormalBolusSolver.fit() (params, curves)
    """

    # ==================================================================
    # STEP 1: ENSURE SVD CACHE EXISTS
    # ==================================================================

    print("\n" + "=" * 70)
    print("STEP 1: SVD FILTERING (create cache if needed)")
    print("=" * 70)

    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])
    cache_dir = get_cache_dir(data_dir, n_svd)

    if use_cache and not force_reprocess:
        is_valid, msg = validate_svd_cache(cache_dir, data_dir, n_svd)

        if is_valid:
            print(f"✓ SVD cache exists and is valid")
            print(f"  Cache directory: {cache_dir}")
        else:
            print(f"✗ {msg}")
            print(f"\n🔄 Running SVD on all {len(all_files)} files...")
            print(f"   This will take 5-10 minutes")
            print(f"   Cache will be ~10-12 GB on disk")
            print(f"   But uses only ~200 MB RAM at a time!\n")

            process_and_cache(data_dir, all_files, n_svd, cache_dir)
            print(f"\n✓ SVD cache created successfully!")
    else:
        if force_reprocess:
            print(f"🔄 Force reprocessing - running SVD on all files...")
            process_and_cache(data_dir, all_files, n_svd,
                              cache_dir if use_cache else None)

    # ==================================================================
    # STEP 2: SELECT ROI
    # ==================================================================

    if roi_coords is None:
        print("\n" + "=" * 70)
        print("STEP 2: INTERACTIVE ROI SELECTION (from SVD cache)")
        print("=" * 70)

        roi_coords = select_roi_from_data(data_dir, n_svd=n_svd)
        print(f"\n✓ ROI selected: {roi_coords}")
    else:
        print("\n" + "=" * 70)
        print("STEP 2: USING PROVIDED ROI")
        print("=" * 70)
        print(f"ROI_COORDS: {roi_coords}")

    # ==================================================================
    # STEP 3: EXTRACT ROI AND PROCESS
    # ==================================================================

    print("\n" + "=" * 70)
    print("STEP 3: EXTRACT ROI + SMOOTH")
    print("=" * 70)

    results = process_all_files(
        data_dir=data_dir,
        n_svd=n_svd,
        roi_coords=roi_coords,
        smooth_window=smooth_window,
        use_cache=use_cache,
        force_reprocess=False,  # SVD already done in step 1
    )

    # ==================================================================
    # STEP 4: FIT LOGNORMAL MODEL (with bolus cache)
    # ==================================================================

    print("\n" + "=" * 70)
    print("STEP 4: FIT LOGNORMAL MODEL")
    print("=" * 70)

    fit_config = {
        'n_iterations': n_iterations,
        'lr': lr,
        'patience': patience,
    }

    bolus_dir = get_bolus_cache_dir(data_dir, n_svd, roi_coords, smooth_window)
    fit_results = None

    if use_cache:
        is_valid, msg = validate_bolus_cache(
            bolus_dir, data_dir, n_svd, roi_coords, smooth_window,
        )
        if is_valid:
            print(f"✓ Bolus cache hit – loading previous fit")
            fit_results = load_bolus_result(bolus_dir)

    if fit_results is None:
        solver = LogNormalBolusSolver()
        fit_results = solver.fit(
            results['time'],
            results['bubbles_smooth'],
            n_iterations=n_iterations,
            lr=lr,
            plot_every=plot_every,
            patience=patience,
            verbose=True,
        )

        if use_cache:
            save_bolus_result(bolus_dir, fit_results)
            save_bolus_cache_metadata(
                bolus_dir, data_dir, n_svd, roi_coords,
                smooth_window, fit_config=fit_config,
            )
            size_mb = get_bolus_cache_size(bolus_dir)
            print(f"✓ Bolus result cached ({size_mb:.2f} MB)")

    # ==================================================================
    # STEP 5: VISUALIZE
    # ==================================================================

    print("\n" + "=" * 70)
    print("STEP 5: VISUALIZE")
    print("=" * 70)

    fig = visualize_fit(
        fit_results,
        title=f"Lognormal Fit - {results['n_files']} files",
    )
    plt.savefig('lognormal_fit_final.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: lognormal_fit_final.png")

    plt.show()

    # ==================================================================
    # SUMMARY
    # ==================================================================

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"ROI used: {results['roi_coords']}")
    print(f"Files processed: {results['n_files']}")
    print(f"Duration: {results['time'][-1]:.1f}s")
    print(f"Enhancement: "
          f"{results['bubbles_smooth'].max() / results['bubbles_smooth'][:50].mean():.2f}x")
    print(f"Peak time: {fit_results['derived']['t_peak']:.2f}s")
    r_squared = 1 - (
        np.var(fit_results['observed'] - fit_results['predicted_curve'])
        / np.var(fit_results['observed'])
    )
    print(f"R²: {r_squared:.4f}")
    print("=" * 70)

    return results, fit_results


# =====================================================================
# CLI entry point
# =====================================================================

if __name__ == '__main__':
    matplotlib.use('TkAgg')

    # -----------------------------------------------------------------
    # CONFIGURATION — edit these to match your setup
    # -----------------------------------------------------------------

    DATA_DIR = r"C:\ultrasoundproject\data\PALA_data_InVivoRatBrainBolus_1\IQ"

    # ROI - Set to None for interactive selection, or specify manually
    ROI_COORDS = None
    # ROI_COORDS = (100, 300, 150, 350)

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

    # -----------------------------------------------------------------
    # RUN
    # -----------------------------------------------------------------

    run_pipeline(
        data_dir=DATA_DIR,
        roi_coords=ROI_COORDS,
        n_svd=N_SVD,
        smooth_window=SMOOTH_WINDOW,
        n_iterations=N_ITERATIONS,
        lr=0.01,
        plot_every=PLOT_EVERY,
        patience=PATIENCE,
        use_cache=USE_CACHE,
        force_reprocess=FORCE_REPROCESS,
    )
