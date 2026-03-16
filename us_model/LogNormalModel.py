"""
CEUS Lognormal Bolus Model Fitting
Processes files with caching, fits lognormal pharmacokinetic model
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from ceus_processor import process_all_files
from select_roi import select_roi_from_data


# ============================================================================
# LOGNORMAL BOLUS MODEL
# ============================================================================

class LogNormalBolusModel(torch.nn.Module):
    """Lognormal pharmacokinetic model for bolus injection"""
    
    def __init__(self, time_points, device='cpu'):
        super().__init__()
        self.device = device
        if not isinstance(time_points, torch.Tensor):
            time_points = torch.tensor(time_points, dtype=torch.float32)
        self.register_buffer('time_points', time_points.float())
    
    def forward(self, params):
        """
        Model: I(t) = I0 + A * lognormal(t - t0; μ, σ)
        
        Parameters:
            I0: Baseline intensity
            A: Amplitude (area under curve)
            t0: Bolus arrival time
            mu: Log-scale mean
            sigma: Log-scale standard deviation
        """
        I0 = params['I0'].squeeze().float()
        A = params['A'].squeeze().float()
        t0 = params['t0'].squeeze().float()
        mu = params['mu'].squeeze().float()
        sigma = params['sigma'].squeeze().float()
        
        t = self.time_points
        device = t.device
        
        # Initialize with baseline
        I = torch.full_like(t, I0.item(), dtype=torch.float32)
        
        # Apply lognormal for t > t0
        mask_bolus = t > t0
        
        if mask_bolus.any():
            t_shifted = torch.clamp(t[mask_bolus] - t0, min=0.001)
            sqrt_2pi = torch.sqrt(torch.tensor(2 * np.pi, dtype=torch.float32, device=device))
            log_t = torch.log(t_shifted)
            exponent = -((log_t - mu) ** 2) / (2 * sigma ** 2)
            lognormal = (1 / (t_shifted * sigma * sqrt_2pi)) * torch.exp(exponent)
            I[mask_bolus] = I0 + A * lognormal
        
        return I


# ============================================================================
# LOGNORMAL SOLVER
# ============================================================================

class LogNormalBolusSolver:
    """Fits lognormal model to time-intensity data"""
    
    def __init__(self, device=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' if device is None else device
        print(f"Solver using: {self.device}")
        if self.device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    def fit(self, time_curve, intensity_curve, n_iterations=500, lr=0.01, 
            plot_every=10, patience=20, verbose=True):
        """
        Fit lognormal model to data
        
        Args:
            time_curve: Time points
            intensity_curve: Observed intensities
            n_iterations: Maximum iterations
            lr: Learning rate
            plot_every: Plot every N iterations (0 to disable)
            patience: Early stopping patience
            verbose: Print progress
            
        Returns:
            Dictionary with fitted parameters and results
        """
        
        t = torch.tensor(time_curve, dtype=torch.float32, device=self.device)
        I_obs = torch.tensor(intensity_curve, dtype=torch.float32, device=self.device)
        
        model = LogNormalBolusModel(time_points=t, device=self.device).to(self.device)
        initial = self._estimate_initial_params(t, I_obs)
        
        if verbose:
            print("\nInitial estimates:")
            for k, v in initial.items():
                print(f"  {k} = {v:.4f}")
        
        # Initialize parameters
        I0 = torch.nn.Parameter(torch.tensor([initial['I0']], dtype=torch.float32, device=self.device))
        A = torch.nn.Parameter(torch.tensor([initial['A']], dtype=torch.float32, device=self.device))
        t0 = torch.nn.Parameter(torch.tensor([initial['t0']], dtype=torch.float32, device=self.device))
        mu = torch.nn.Parameter(torch.tensor([initial['mu']], dtype=torch.float32, device=self.device))
        sigma = torch.nn.Parameter(torch.tensor([initial['sigma']], dtype=torch.float32, device=self.device))
        
        optimizer = torch.optim.Adam([I0, A, t0, mu, sigma], lr=lr)
        
        losses = []
        best_loss = float('inf')
        no_improve = 0
        
        if verbose:
            print(f"\nOptimizing (max {n_iterations} iter, patience={patience})...\n")
        
        if plot_every > 0:
            plt.ion()
            fig_progress, ax_progress = plt.subplots(1, 1, figsize=(14, 6))
        
        pbar = tqdm(range(n_iterations), desc="Fitting", ncols=100)
        
        for iteration in pbar:
            optimizer.zero_grad()
            
            # Enforce constraints
            params = {
                'I0': torch.abs(I0),
                'A': torch.abs(A),
                't0': t0,
                'mu': mu,
                'sigma': torch.abs(sigma) + 0.01
            }
            
            I_pred = model(params)
            loss = torch.nn.functional.mse_loss(I_pred, I_obs)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            losses.append(current_loss)
            
            # Early stopping check
            if current_loss < best_loss - 1e-6:
                best_loss = current_loss
                no_improve = 0
            else:
                no_improve += 1
            
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'patience': f'{no_improve}/{patience}'})
            
            if no_improve >= patience:
                print(f"\n✓ Early stopping at iteration {iteration}")
                pbar.close()
                break
            
            # Progress plot
            if plot_every > 0 and (iteration % plot_every == 0 or iteration == n_iterations - 1):
                with torch.no_grad():
                    ax_progress.clear()
                    ax_progress.plot(t.cpu().numpy(), I_obs.cpu().numpy(), 'o', 
                                   alpha=0.3, markersize=2, label='Observed', color='blue')
                    ax_progress.plot(t.cpu().numpy(), I_pred.cpu().numpy(), '-', 
                                   linewidth=2, label='Fit', color='red')
                    ax_progress.set_xlabel('Time (s)')
                    ax_progress.set_ylabel('Intensity')
                    ax_progress.set_title(f'Iter {iteration}/{n_iterations} - Loss: {current_loss:.4f}')
                    ax_progress.legend()
                    ax_progress.grid(True, alpha=0.3)
                    plt.pause(0.01)
        
        if iteration == n_iterations - 1:
            pbar.close()
        
        if plot_every > 0:
            plt.ioff()
            plt.close(fig_progress)
        
        # Extract final parameters
        final_params = {
            'I0': torch.abs(I0).item(),
            'A': torch.abs(A).item(),
            't0': t0.item(),
            'mu': mu.item(),
            'sigma': (torch.abs(sigma) + 0.01).item()
        }
        
        derived = self._compute_derived(final_params)
        
        # Final prediction
        with torch.no_grad():
            params_final = {
                'I0': torch.abs(I0),
                'A': torch.abs(A),
                't0': t0,
                'mu': mu,
                'sigma': torch.abs(sigma) + 0.01
            }
            I_final = model(params_final)
        
        if verbose:
            print("\n" + "="*60)
            print("FITTED PARAMETERS")
            print("="*60)
            for k, v in final_params.items():
                print(f"  {k:8s} = {v:10.4f}")
            print("\nDERIVED")
            print("-"*60)
            for k, v in derived.items():
                print(f"  {k:12s} = {v:10.4f}")
            print("="*60)
        
        return {
            'params': final_params,
            'derived': derived,
            'predicted_curve': I_final.cpu().numpy(),
            'losses': losses,
            'time': t.cpu().numpy(),
            'observed': I_obs.cpu().numpy()
        }
    
    def _estimate_initial_params(self, t, I_obs):
        """Estimate initial parameter values"""
        I0 = I_obs[:50].mean().item()
        I_max = I_obs.max().item()
        t_peak_idx = I_obs.argmax().item()
        t_peak = t[t_peak_idx].item()
        A = (I_max - I0) * (t_peak - t[0].item()) * 2.5
        
        threshold = I0 + 0.05 * (I_max - I0)
        above = torch.where(I_obs > threshold)[0]
        t0 = t[above[0]].item() if len(above) > 0 else t[0].item()
        
        sigma = 0.6
        mu = np.log(max(t_peak - t0, 0.5)) + sigma**2
        
        return {'I0': I0, 'A': max(A, 1.0), 't0': t0, 'mu': mu, 'sigma': sigma}
    
    def _compute_derived(self, p):
        """Compute derived parameters"""
        t_peak = p['t0'] + np.exp(p['mu'] - p['sigma']**2)
        MTT = p['t0'] + np.exp(p['mu'] + p['sigma']**2 / 2)
        return {'t_peak': t_peak, 'MTT': MTT, 'AUC': p['A']}


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_fit(results, title="Lognormal Fit"):
    """Create comprehensive fit visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    time = results['time']
    obs = results['observed']
    pred = results['predicted_curve']
    params = results['params']
    derived = results['derived']
    
    # 1. Main fit
    ax = axes[0, 0]
    ax.plot(time, obs, 'o', alpha=0.3, markersize=2, label='Observed')
    ax.plot(time, pred, '-', linewidth=2, label='Fitted', color='red')
    ax.axvline(params['t0'], color='green', linestyle='--', alpha=0.5, label=f"t0={params['t0']:.1f}s")
    ax.axvline(derived['t_peak'], color='orange', linestyle='--', alpha=0.5, label=f"peak={derived['t_peak']:.1f}s")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Intensity')
    ax.set_title('Fit')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. Residuals
    ax = axes[0, 1]
    residuals = obs - pred
    ax.plot(time, residuals, 'o', markersize=2, alpha=0.6)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.fill_between(time, -residuals.std(), residuals.std(), alpha=0.2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Residual')
    ax.set_title(f'Residuals (σ={residuals.std():.2f})')
    ax.grid(True, alpha=0.3)
    
    # 3. Loss
    ax = axes[0, 2]
    if len(results['losses']) > 0:
        ax.plot(results['losses'], linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # 4. Wash-in detail
    ax = axes[1, 0]
    t0_idx = np.argmin(np.abs(time - params['t0']))
    peak_idx = np.argmin(np.abs(time - derived['t_peak']))
    start = max(0, t0_idx - 100)
    end = min(len(time), peak_idx + 200)
    ax.plot(time[start:end], obs[start:end], 'o', alpha=0.4, markersize=3, label='Observed')
    ax.plot(time[start:end], pred[start:end], '-', linewidth=2, label='Fitted', color='red')
    ax.axvline(params['t0'], color='green', linestyle='--', alpha=0.5)
    ax.axvline(derived['t_peak'], color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Intensity')
    ax.set_title('Wash-in Detail')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 5. Log scale
    ax = axes[1, 1]
    I_base_removed = obs - params['I0']
    I_pred_removed = pred - params['I0']
    mask_pos = I_base_removed > 0
    ax.semilogy(time[mask_pos], I_base_removed[mask_pos], 'o', alpha=0.4, markersize=2, label='Observed')
    mask_pred = I_pred_removed > 0
    ax.semilogy(time[mask_pred], I_pred_removed[mask_pred], '-', linewidth=2, label='Fitted', color='red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Intensity - Baseline (log)')
    ax.set_title('Log Scale')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')
    summary = "FITTED PARAMETERS\n" + "="*40 + "\n\n"
    summary += f"{'I0':<12} {params['I0']:>12.2f}\n"
    summary += f"{'A':<12} {params['A']:>12.2f}\n"
    summary += f"{'t0':<12} {params['t0']:>12.3f} s\n"
    summary += f"{'μ':<12} {params['mu']:>12.3f}\n"
    summary += f"{'σ':<12} {params['sigma']:>12.3f}\n"
    summary += "\n" + "="*40 + "\nDERIVED\n" + "="*40 + "\n\n"
    summary += f"{'t_peak':<20} {derived['t_peak']:>8.3f} s\n"
    summary += f"{'MTT':<20} {derived['MTT']:>8.3f} s\n"
    summary += f"{'AUC':<20} {derived['AUC']:>8.2f}\n"
    summary += "\n" + "="*40 + "\nQUALITY\n" + "="*40 + "\n\n"
    if len(results['losses']) > 0:
        summary += f"{'Final loss':<20} {results['losses'][-1]:>8.6f}\n"
    summary += f"{'Residual σ':<20} {residuals.std():>8.3f}\n"
    summary += f"{'R²':<20} {1 - np.var(residuals)/np.var(obs):>8.4f}\n"
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN SCRIPT
# ============================================================================
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
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
    
    # ========================================================================
    # STEP 1: ENSURE SVD CACHE EXISTS
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 1: SVD FILTERING (create cache if needed)")
    print("="*70)
    
    from cache_utils import get_cache_dir, validate_svd_cache  # ← שנה כאן!
    from ceus_processor import process_and_cache
    
    all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.mat')])
    cache_dir = get_cache_dir(DATA_DIR, N_SVD)  # ← שנה כאן!
    
    # Check if SVD cache exists
    if USE_CACHE and not FORCE_REPROCESS:
        is_valid, msg = validate_svd_cache(cache_dir, DATA_DIR, N_SVD)  # ← cache_dir במקום cache_file
        
        if is_valid:
            print(f"✓ SVD cache exists and is valid")
            print(f"  Cache directory: {cache_dir}")
        else:
            print(f"✗ {msg}")
            print(f"\n🔄 Running SVD on all {len(all_files)} files...")
            print(f"   This will take 5-10 minutes")
            print(f"   Cache will be ~10-12 GB on disk")
            print(f"   But uses only ~200 MB RAM at a time!\n")
            
            # Run SVD on all files (one at a time - memory efficient!)
            params = process_and_cache(DATA_DIR, all_files, N_SVD, cache_dir)  # ← cache_dir
            
            print(f"\n✓ SVD cache created successfully!")
    else:
        if FORCE_REPROCESS:
            print(f"🔄 Force reprocessing - running SVD on all files...")
            params = process_and_cache(DATA_DIR, all_files, N_SVD, cache_dir if USE_CACHE else None)
    
    # ========================================================================
    # STEP 2: SELECT ROI (from SVD cache)
    # ========================================================================
    
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
    
    # ========================================================================
    # STEP 3: EXTRACT ROI AND PROCESS
    # ========================================================================
    
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
    
    # ========================================================================
    # STEP 4: FIT LOGNORMAL MODEL
    # ========================================================================
    
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
    
    # ========================================================================
    # STEP 5: VISUALIZE
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 5: VISUALIZE")
    print("="*70)
    
    fig = visualize_fit(fit_results, title=f"Lognormal Fit - {results['n_files']} files")
    plt.savefig('lognormal_fit_final.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: lognormal_fit_final.png")
    
    plt.show()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
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