"""
qCEUS-Net: Joint Parameter Fitting and Registration for CEUS Bolus Data
Based on qDIVIM architecture adapted for contrast-enhanced ultrasound
"""

import os
import sys
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
import torch.nn as nn
import torch
import numpy as np


class CEUS_Bolus_Fitting_Net(nn.Module):
    """
    Fitting sub-network for lognormal bolus parameters
    Predicts spatial maps of: I0, A, t0, μ, σ
    """

    def __init__(self,
                 inshape,
                 time_points,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 dropout=0.1,
                 normalize_output=True):
        """
        Parameters:
            inshape: Spatial dimensions (nx, ny)
            time_points: Time sampling points [n_frames]
            nb_unet_features: Unet convolutional features
            normalize_output: Whether to normalize by baseline
        """
        super().__init__()
        
        # Store time points as tensor
        if isinstance(time_points, np.ndarray):
            time_points = torch.from_numpy(time_points).float()
        self.time_points = time_points
        self.n_frames = len(time_points)
        self.normalize_output = normalize_output
        
        # Dimensionality
        ndims = len(inshape)
        assert ndims == 2, 'CEUS processing expects 2D spatial dimensions'
        
        # Core U-Net (encoder)
        self.unet_model = vxm.networks.Unet(
            inshape=inshape,
            infeats=self.n_frames,  # Time frames as input channels
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=False,
        )
        
        # Parameter prediction layers (5 decoders)
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.I0_layer = Conv(self.unet_model.final_nf, 1, kernel_size=3, padding=1)
        self.A_layer = Conv(self.unet_model.final_nf, 1, kernel_size=3, padding=1)
        self.t0_layer = Conv(self.unet_model.final_nf, 1, kernel_size=3, padding=1)
        self.mu_layer = Conv(self.unet_model.final_nf, 1, kernel_size=3, padding=1)
        self.sigma_layer = Conv(self.unet_model.final_nf, 1, kernel_size=3, padding=1)
        
        # Parameter bounds [min, max]
        # [I0_min, A_min, t0_min, mu_min, sigma_min]
        # [I0_max, A_max, t0_max, mu_max, sigma_max]
        bounds = torch.tensor([
            [0.0,    0.0,   -5.0,  -2.0,  0.1],   # Min bounds
            [500.0,  20000.0, 20.0,  5.0,  3.0]   # Max bounds
        ])
        
        # Expand bounds slightly for sigmoid linearity
        boundsrange = 0.15 * (bounds[1] - bounds[0])
        self.bounds = torch.stack((
            bounds[0] - boundsrange, 
            bounds[1] + boundsrange
        ))
        
        # Store bounds as buffer (moves with model to device)
        self.register_buffer('bounds_buffer', self.bounds)

    def lognormal_bolus_model(self, I0, A, t0, mu, sigma):
        """
        Forward model: Lognormal bolus
        
        I(t) = I0 + A * (1/((t-t0)*σ*√(2π))) * exp(-[ln(t-t0) - μ]²/(2σ²))
        
        Args:
            I0, A, t0, mu, sigma: Parameter maps [B, 1, nx, ny]
            
        Returns:
            recon: Reconstructed images [B, n_frames, nx, ny]
        """
        B, _, nx, ny = I0.shape
        device = I0.device
        
        # Time points
        t = self.time_points.to(device)  # [n_frames]
        
        # Reshape parameters for broadcasting
        I0 = I0.view(B, 1, nx, ny)      # [B, 1, nx, ny]
        A = A.view(B, 1, nx, ny)
        t0 = t0.view(B, 1, nx, ny)
        mu = mu.view(B, 1, nx, ny)
        sigma = sigma.view(B, 1, nx, ny)
        
        # Expand time: [n_frames] -> [B, n_frames, nx, ny]
        t = t.view(1, -1, 1, 1).expand(B, -1, nx, ny)
        
        # Shifted time
        t_shifted = t - t0  # [B, n_frames, nx, ny]
        
        # Mask for t > t0 (bolus hasn't arrived yet)
        mask = (t_shifted > 0.001).float()  # Small epsilon to avoid log(0)
        t_shifted = torch.clamp(t_shifted, min=0.001)
        
        # Lognormal PDF
        sqrt_2pi = np.sqrt(2 * np.pi)
        log_t = torch.log(t_shifted)
        
        exponent = -((log_t - mu) ** 2) / (2 * sigma ** 2)
        lognormal = (1 / (t_shifted * sigma * sqrt_2pi)) * torch.exp(exponent)
        
        # Apply mask and add baseline
        recon = I0 + A * lognormal * mask  # [B, n_frames, nx, ny]
        
        return recon

    def forward(self, img):
        """
        Args:
            img: Input CEUS sequence [B, n_frames, nx, ny]
            
        Returns:
            recon: Model-reconstructed sequence
            I0, A, t0, mu, sigma: Parameter maps
        """
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        
        B, n_frames, nx, ny = img.shape
        
        # U-Net feature extraction
        x = self.unet_model(img)
        
        # Parameter prediction (unconstrained)
        I0_raw = self.I0_layer(x)
        A_raw = self.A_layer(x)
        t0_raw = self.t0_layer(x)
        mu_raw = self.mu_layer(x)
        sigma_raw = self.sigma_layer(x)
        
        # Apply bounds using sigmoid
        bounds = self.bounds_buffer
        
        I0 = bounds[0, 0] + torch.sigmoid(I0_raw) * (bounds[1, 0] - bounds[0, 0])
        A = bounds[0, 1] + torch.sigmoid(A_raw) * (bounds[1, 1] - bounds[0, 1])
        t0 = bounds[0, 2] + torch.sigmoid(t0_raw) * (bounds[1, 2] - bounds[0, 2])
        mu = bounds[0, 3] + torch.sigmoid(mu_raw) * (bounds[1, 3] - bounds[0, 3])
        sigma = bounds[0, 4] + torch.sigmoid(sigma_raw) * (bounds[1, 4] - bounds[0, 4])
        
        # Forward model: generate reconstructed images
        recon = self.lognormal_bolus_model(I0, A, t0, mu, sigma)
        
        if self.normalize_output:
            # Normalize by baseline (first frame)
            baseline = img[:, 0:1, :, :].clone()
            recon = recon / (baseline + 1e-6) * baseline.mean()
        
        return recon, I0, A, t0, mu, sigma


class Registration_SubNet(nn.Module):
    """
    Registration sub-network using VoxelMorph
    Aligns raw images to model-reconstructed images
    """
    
    def __init__(self, inshape, n_frames, nb_unet_features=None, 
                 int_steps=7, int_downsize=2):
        super().__init__()
        self.inshape = inshape
        self.n_frames = n_frames
        
        # VoxelMorph registration network
        self.model_vxm = vxm.networks.VxmDense(
            inshape=inshape,
            nb_unet_features=nb_unet_features,
            int_steps=int_steps,
            int_downsize=int_downsize,
        )

    def forward(self, moving, fixed):
        """
        Args:
            moving: Raw acquired images [B*n_frames, nx, ny, 1]
            fixed: Model-reconstructed images [B*n_frames, nx, ny, 1]
            
        Returns:
            warped: Warped moving images
            flow_pre_integrated: Flow field before integration
            flow_integrated: Final integrated flow field
        """
        # VoxelMorph expects [B, C, H, W] format
        moving = moving.float().permute(0, -1, 1, 2)  # [B*n_frames, 1, nx, ny]
        fixed = fixed.to(moving.device).float().permute(0, -1, 1, 2)
        
        # Registration
        warped, flow = self.model_vxm(moving, fixed)
        
        return warped, flow, flow  # Return flow twice for compatibility


class qCEUS_Net(nn.Module):
    """
    Complete qCEUS network: Joint fitting and registration
    
    This network simultaneously:
    1. Predicts lognormal bolus parameters (I0, A, t0, μ, σ) for each pixel
    2. Registers raw images to model-predicted images
    """

    def __init__(self,
                 inshape,
                 time_points,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 dropout=0.1,
                 nb_unet_features_reg=None,
                 int_steps=7,
                 int_downsize=2,
                 register_baseline=False):
        """
        Parameters:
            inshape: Spatial dimensions (nx, ny)
            time_points: Time sampling [n_frames]
            nb_unet_features: Features for fitting network
            nb_unet_features_reg: Features for registration network
            int_steps: Integration steps for flow field
            register_baseline: Whether to register first frame
        """
        super().__init__()
        
        self.time_points = time_points
        self.n_frames = len(time_points)
        self.register_baseline = register_baseline
        
        # Fitting sub-network
        self.fitting_net = CEUS_Bolus_Fitting_Net(
            inshape=inshape,
            time_points=time_points,
            nb_unet_features=nb_unet_features,
            nb_unet_levels=nb_unet_levels,
            unet_feat_mult=unet_feat_mult,
            nb_unet_conv_per_level=nb_unet_conv_per_level,
            dropout=dropout
        )
        
        # Registration sub-network
        self.reg_net = Registration_SubNet(
            inshape=inshape,
            n_frames=self.n_frames,
            nb_unet_features=nb_unet_features_reg,
            int_steps=int_steps,
            int_downsize=int_downsize
        )

    def forward(self, img):
        """
        Args:
            img: Input CEUS sequence [B, n_frames, nx, ny]
            
        Returns:
            warped: Registered images [B, n_frames, nx, ny]
            flow_pre_integrated: Deformation fields before integration
            recon: Model-reconstructed images [B, n_frames, nx, ny]
            params: Dictionary with parameter maps
            flow_integrated: Final flow fields
        """
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        
        B, n_frames, nx, ny = img.shape
        
        # Step 1: Fitting network predicts parameters and reconstructs images
        recon, I0, A, t0, mu, sigma = self.fitting_net(img)
        
        # Step 2: Registration - align raw images to model predictions
        if not self.register_baseline:
            # Skip first frame (baseline)
            recon_moving = recon[:, 1:, :, :].reshape(-1, nx, ny).unsqueeze(-1)
            img_moving = img[:, 1:, :, :].reshape(-1, nx, ny).unsqueeze(-1)
            
            # Register
            warped_frames, flow_pre, flow_post = self.reg_net(img_moving, recon_moving)
            
            # Reshape back
            warped_frames = warped_frames.view(B, n_frames - 1, nx, ny)
            
            # Add back baseline (unregistered)
            baseline = img[:, 0:1, :, :]
            warped = torch.cat([baseline, warped_frames], dim=1)
            
            # Flow fields
            flow_pre = flow_pre.view(B, n_frames - 1, 2, nx, ny)
            flow_post = flow_post.view(B, n_frames - 1, 2, nx, ny)
            
            # Add zero flow for baseline
            zero_flow = torch.zeros(B, 1, 2, nx, ny, device=img.device)
            flow_pre_integrated = torch.cat([zero_flow, flow_pre], dim=1)
            flow_integrated = torch.cat([zero_flow, flow_post], dim=1)
            
        else:
            # Register all frames including baseline
            recon_moving = recon.reshape(B * n_frames, nx, ny).unsqueeze(-1)
            img_moving = img.reshape(B * n_frames, nx, ny).unsqueeze(-1)
            
            warped_frames, flow_pre, flow_post = self.reg_net(img_moving, recon_moving)
            
            warped = warped_frames.permute(1, 0, 2, 3)  # [B, n_frames, nx, ny]
            flow_pre_integrated = flow_pre.unsqueeze(0)
            flow_integrated = flow_post.unsqueeze(0)
        
        # Package parameter maps
        params = {
            'I0': I0,
            'A': A,
            't0': t0,
            'mu': mu,
            'sigma': sigma
        }
        
        return warped, flow_pre_integrated, recon, params, flow_integrated


# ============================================================
# USAGE EXAMPLE
# ============================================================

def example_qceus_net():
    """
    Example of using qCEUS-Net
    """
    print("="*60)
    print("qCEUS-Net: Joint Fitting and Registration")
    print("="*60)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Data dimensions
    B = 2           # Batch size
    n_frames = 50   # Number of time frames
    nx, ny = 64, 64 # Spatial dimensions
    
    # Time points (e.g., 50 frames over 30 seconds)
    time_points = np.linspace(0, 30, n_frames)
    
    # Create synthetic CEUS data
    print(f"\nGenerating synthetic data:")
    print(f"  Shape: [{B}, {n_frames}, {nx}, {ny}]")
    print(f"  Time: {time_points[0]:.1f} - {time_points[-1]:.1f} s")
    
    # Simulate bolus curves for each pixel
    img = torch.zeros(B, n_frames, nx, ny, device=device)
    
    for b in range(B):
        for i in range(nx):
            for j in range(ny):
                # Random parameters for each pixel
                I0_true = 100 + 20 * np.random.randn()
                A_true = 5000 + 1000 * np.random.randn()
                t0_true = 5 + 2 * np.random.randn()
                mu_true = 2.0 + 0.5 * np.random.randn()
                sigma_true = 0.8 + 0.2 * np.random.randn()
                
                # Generate lognormal bolus
                t = time_points
                t_shifted = np.maximum(t - t0_true, 0.001)
                log_t = np.log(t_shifted)
                lognormal = (1 / (t_shifted * sigma_true * np.sqrt(2*np.pi))) * \
                           np.exp(-((log_t - mu_true)**2) / (2*sigma_true**2))
                
                # Apply mask
                mask = (t > t0_true).astype(float)
                intensity = I0_true + A_true * lognormal * mask
                
                img[b, :, i, j] = torch.tensor(intensity, dtype=torch.float32)
    
    # Add noise
    img = img + 50 * torch.randn_like(img)
    
    # Initialize network
    print("\nInitializing qCEUS-Net...")
    model = qCEUS_Net(
        inshape=(nx, ny),
        time_points=time_points,
        nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]],
        nb_unet_features_reg=[[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]],
        int_steps=7,
        register_baseline=False
    ).to(device)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    print("\nForward pass...")
    with torch.no_grad():
        warped, flow_pre, recon, params, flow_post = model(img)
    
    print("\nOutput shapes:")
    print(f"  Warped images: {warped.shape}")
    print(f"  Reconstructed: {recon.shape}")
    print(f"  Flow fields: {flow_post.shape}")
    print(f"  Parameter maps:")
    for k, v in params.items():
        print(f"    {k}: {v.shape}")
    
    # Visualize results for one sample
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    
    sample_idx = 0
    mid_frame = n_frames // 2
    
    # Row 1: Parameter maps
    im0 = axes[0, 0].imshow(params['I0'][sample_idx, 0].cpu(), cmap='gray')
    axes[0, 0].set_title('I0 (Baseline)')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(params['A'][sample_idx, 0].cpu(), cmap='hot')
    axes[0, 1].set_title('A (Amplitude)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(params['t0'][sample_idx, 0].cpu(), cmap='viridis')
    axes[0, 2].set_title('t0 (Arrival)')
    plt.colorbar(im2, ax=axes[0, 2])
    
    im3 = axes[0, 3].imshow(params['mu'][sample_idx, 0].cpu(), cmap='viridis')
    axes[0, 3].set_title('μ (Location)')
    plt.colorbar(im3, ax=axes[0, 3])
    
    im4 = axes[0, 4].imshow(params['sigma'][sample_idx, 0].cpu(), cmap='viridis')
    axes[0, 4].set_title('σ (Dispersion)')
    plt.colorbar(im4, ax=axes[0, 4])
    
    # Row 2: Images at mid-frame
    axes[1, 0].imshow(img[sample_idx, mid_frame].cpu(), cmap='gray')
    axes[1, 0].set_title(f'Original (t={time_points[mid_frame]:.1f}s)')
    
    axes[1, 1].imshow(recon[sample_idx, mid_frame].cpu().detach(), cmap='gray')
    axes[1, 1].set_title('Reconstructed')
    
    axes[1, 2].imshow(warped[sample_idx, mid_frame].cpu(), cmap='gray')
    axes[1, 2].set_title('Registered')
    
    diff = (img[sample_idx, mid_frame] - recon[sample_idx, mid_frame].detach()).cpu()
    axes[1, 3].imshow(diff, cmap='RdBu_r', vmin=-diff.abs().max(), vmax=diff.abs().max())
    axes[1, 3].set_title('Residual')
    
    # Flow magnitude
    flow_mag = torch.sqrt((flow_post[sample_idx, mid_frame, 0]**2 + 
                          flow_post[sample_idx, mid_frame, 1]**2)).cpu()
    axes[1, 4].imshow(flow_mag, cmap='hot')
    axes[1, 4].set_title('Flow Magnitude')
    
    # Row 3: Time curves for central pixel
    center_i, center_j = nx // 2, ny // 2
    
    axes[2, 0].plot(time_points, img[sample_idx, :, center_i, center_j].cpu(), 
                    'o-', label='Original', alpha=0.6)
    axes[2, 0].plot(time_points, recon[sample_idx, :, center_i, center_j].cpu().detach(), 
                    '-', label='Model', linewidth=2)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Intensity')
    axes[2, 0].set_title('Center Pixel Curve')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Hide remaining subplots
    for i in range(1, 5):
        axes[2, i].axis('off')
    
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('qceus_net_output.png', dpi=300, bbox_inches='tight')
    print("\n✓ Results saved to 'qceus_net_output.png'")
    plt.show()


if __name__ == '__main__':
    example_qceus_net()