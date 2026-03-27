"""
CUDA Utilities
"""

import os
import torch
import warnings

def setup_cuda():
    """Setup CUDA with proper error handling."""
    # Suppress CUDA warnings
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return False
    
    try:
        # Test CUDA availability
        torch.cuda.empty_cache()
        device_count = torch.cuda.device_count()
        if device_count > 0:
            print(f"CUDA available with {device_count} device(s)")
            return True
        else:
            print("No CUDA devices found")
            return False
    except RuntimeError as e:
        if "CUDA" in str(e) and ("busy" in str(e) or "unavailable" in str(e)):
            print("CUDA is busy/unavailable, falling back to CPU")
            return False
        else:
            print(f"CUDA error: {e}")
            return False

def get_best_device():
    """Get the best available device."""
    if setup_cuda():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def suppress_cuda_warnings():
    """Suppress CUDA warnings."""
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
    warnings.filterwarnings("ignore", message=".*CUDA.*")
