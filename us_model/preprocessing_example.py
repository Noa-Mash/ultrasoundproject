"""
Process CEUS data - works with single file OR entire folder

INSTRUCTIONS:
1. Edit DATA_PATH below (can be a .mat file OR a folder)
2. Run: python process_ceus.py
"""

from pathlib import Path
from data_loader import load_ceus_data
from filters import filter_both

# ============================================================
# EDIT THIS PATH - can be a file OR folder
# ============================================================

DATA_PATH = "PALA/PALA_data_InVivoRatBrain/IQ"
# Examples:
#   Single file: "/home/user/scan.mat"
#   Folder:      "/home/user/data_folder"

N_COMPONENTS = 50  # ← ADJUST IF NEEDED

# ============================================================


def process_single_file(mat_file: str, n_components: int = 50) -> dict:
    """Process a single .mat file and return structured data."""
    
    print(f"\nProcessing: {Path(mat_file).name}")
    print("-"*60)
    
    # Load
    data = load_ceus_data(mat_file)
    IQ_magnitude = data['IQ_magnitude']
    params = data['params']
    
    # Filter
    tissue, bubbles = filter_both(IQ_magnitude, n_components=n_components)
    
    # Build per-frame dictionary
    z, x, t = tissue.shape
    frames = {}
    for frame_idx in range(t):
        frames[frame_idx] = {
            'IQ_magnitude': IQ_magnitude[:, :, frame_idx],
            'tissue': tissue[:, :, frame_idx],
            'bubbles': bubbles[:, :, frame_idx]
        }
    
    return {
        'IQ_magnitude': IQ_magnitude,
        'tissue': tissue,
        'bubbles': bubbles,
        'params': params,
        'frames': frames,
        'filename': Path(mat_file).name
    }


def main():
    print("="*60)
    print("CEUS Data Processing")
    print("="*60)
    print(f"Input: {DATA_PATH}")
    print(f"SVD components: {N_COMPONENTS}")
    print()
    
    path = Path(DATA_PATH)
    
    # Check if path exists
    if not path.exists():
        print(f"ERROR: Path does not exist: {DATA_PATH}")
        print("Please edit DATA_PATH in this script (line 15)")
        return None
    
    # Single file
    if path.is_file() and path.suffix == '.mat':
        print("Mode: Single file")
        data = process_single_file(str(path), n_components=N_COMPONENTS)
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"Shape: {data['tissue'].shape}")
        print(f"Frames: {data['tissue'].shape[2]}")
        print()
        print("Access your data:")
        print("  data['tissue']       - full tissue array [z,x,t]")
        print("  data['bubbles']      - full bubbles array [z,x,t]")
        print("  data['frames'][10]   - frame 10")
        print("  data['params']       - acquisition parameters")
        
        return data
    
    # Folder with multiple files
    elif path.is_dir():
        print("Mode: Folder (batch processing)")
        
        mat_files = list(path.glob('*.mat'))
        if len(mat_files) == 0:
            print(f"ERROR: No .mat files found in {DATA_PATH}")
            return {}
        
        print(f"Found {len(mat_files)} .mat files")
        print()
        
        all_data = {}
        for i, mat_file in enumerate(mat_files, 1):
            print(f"[{i}/{len(mat_files)}]", end=" ")
            
            try:
                data = process_single_file(str(mat_file), n_components=N_COMPONENTS)
                all_data[mat_file.stem] = data
                print(f"  ✓ Success: {data['tissue'].shape}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        print("\n" + "="*60)
        print("COMPLETE!")
        print("="*60)
        print(f"Successfully processed: {len(all_data)}/{len(mat_files)} files")
        print()
        print("Access your data:")
        print("  all_data['filename']['tissue']   - tissue for that file")
        print("  all_data['filename']['bubbles']  - bubbles for that file")
        print("  all_data['filename']['frames'][i] - specific frame")
        
        if len(all_data) > 0:
            first_key = list(all_data.keys())[0]
            print()
            print("Example:")
            print(f"  data = all_data['{first_key}']")
            print(f"  tissue = data['tissue']  # shape: {all_data[first_key]['tissue'].shape}")
        
        return all_data
    
    else:
        print(f"ERROR: Path must be a .mat file or a folder")
        print(f"Got: {DATA_PATH}")
        return None


if __name__ == '__main__':
    # Check if path was updated
    if DATA_PATH == "path/to/your/data":
        print("Please edit DATA_PATH in this script first!")
        print("Open process_ceus.py and change line 15")
        print()
        print("Examples:")
        print('  Single file: DATA_PATH = "/home/user/scan.mat"')
        print('  Folder:      DATA_PATH = "/home/user/data_folder"')
    else:
        data = main()