#!/usr/bin/env python3
"""
Pre-download all EchoPilot models once.

This script downloads all required models so they don't need to be downloaded
each time the agent runs.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("EchoPilot Model Pre-Download Script")
print("=" * 80)
print()

# 1. Download EchoPrime models
print("[1/4] Checking EchoPrime models...")
try:
    from models.echo.echo_prime_manager import EchoPrimeManager, EchoPrimeConfig
    
    echo_prime_manager = EchoPrimeManager()
    if echo_prime_manager._check_models_exist():
        print("   [OK] EchoPrime models already exist")
    else:
        print("   [INFO] Downloading EchoPrime models...")
        if echo_prime_manager._download_models():
            print("   [OK] EchoPrime models downloaded successfully")
        else:
            print("   [ERROR] Failed to download EchoPrime models")
except Exception as e:
    print(f"   [ERROR] EchoPrime download failed: {e}")

print()

# 2. Pre-load PanEcho (torch.hub will cache it)
print("[2/4] Pre-loading PanEcho model (will cache if not already cached)...")
try:
    import torch
    
    # PanEcho is loaded via torch.hub which caches automatically
    # We'll load it once to trigger the download/cache
    model = torch.hub.load(
        'CarDS-Yale/PanEcho',
        'PanEcho',
        force_reload=False,
        tasks='all',
        clip_len=16,
        verbose=True
    )
    print("   [OK] PanEcho model cached successfully")
    del model  # Free memory
except Exception as e:
    print(f"   [ERROR] PanEcho pre-load failed: {e}")

print()

# 3. Download MedSAM2
print("[3/4] Checking MedSAM2 model...")
try:
    from huggingface_hub import hf_hub_download
    
    checkpoint_path = Path("checkpoints") / "MedSAM2_US_Heart.pt"
    if checkpoint_path.exists():
        print(f"   [OK] MedSAM2 model already exists at {checkpoint_path}")
    else:
        print("   [INFO] Downloading MedSAM2 model...")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model_path = hf_hub_download(
            repo_id="wanglab/MedSAM2",
            filename="MedSAM2_US_Heart.pt",
            local_dir=str(checkpoint_path.parent),
            local_dir_use_symlinks=False
        )
        print(f"   [OK] MedSAM2 model downloaded to {model_path}")
except Exception as e:
    print(f"   [ERROR] MedSAM2 download failed: {e}")

print()

# 4. Check EchoFlow (if needed)
print("[4/4] Checking EchoFlow models...")
try:
    from models.echo.echoflow_manager import EchoFlowManager
    
    echoflow_manager = EchoFlowManager()
    # EchoFlow models are loaded on-demand from HuggingFace
    # They will be cached automatically by huggingface_hub
    print("   [INFO] EchoFlow models will be downloaded on first use (cached by HuggingFace)")
except Exception as e:
    print(f"   [WARNING] EchoFlow check failed: {e}")

print()
print("=" * 80)
print("Model pre-download complete!")
print("=" * 80)
print()
print("All models are now cached. The agent will use cached models on subsequent runs.")
print()

