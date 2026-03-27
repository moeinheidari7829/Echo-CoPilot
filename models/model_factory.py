"""
Model Factory

This module provides a factory for loading and managing different types of models
using real weights from the model_weights directory.
"""

import os
import sys
import torch
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelFactory:
    """Factory for loading and managing different types of models."""

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self.model_weights_dir = Path("model_weights")
        self.checkpoints_dir = Path("checkpoints")

    def load_echo_prime(self) -> Optional[Any]:
        """Load EchoPrime model with real weights."""
        try:
            # Add echo_prime to path
            echo_prime_path = self.model_weights_dir / "echo_prime"
            if echo_prime_path.exists():
                # Add echo_prime directory to path so it can find utils.py
                sys.path.insert(0, str(echo_prime_path))

                # Import EchoPrime model
                # from model import EchoPrime
                # from tool_repos.EchoPrime-main.echo_prime.model import EchoPrime
                from tool_repos.EchoPrime1.echo_prime.model import EchoPrime

                # Load model - this will automatically load all weights
                model = EchoPrime(device="cuda" if torch.cuda.is_available() else "cpu")

                print("[OK] EchoPrime model loaded successfully")
                return model
            else:
                print(f"[ERROR] EchoPrime weights directory not found: {echo_prime_path}")
                return None

        except Exception as e:
            print(f"[ERROR] Failed to load EchoPrime model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_panecho(self) -> Optional[Any]:
        """Load PanEcho model with real weights and all available tasks."""
        try:
            # Load PanEcho from torch hub with all tasks (default behavior)
            # force_reload=False ensures models are cached and not re-downloaded
            # trust_repo=True to avoid prompts (repo is trusted)
            model = torch.hub.load(
                'CarDS-Yale/PanEcho',
                'PanEcho',
                force_reload=False,  # Use cached version if available
                trust_repo=True,  # Trust the repository to avoid prompts
                tasks='all',  # Use all available tasks
                clip_len=16,
                verbose=False  # Reduce verbosity after first download
            )
            model.eval()

            # Move to appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            # Get the list of available tasks
            available_tasks = list(model.tasks) if hasattr(model, 'tasks') else []
            task_names = [task.task_name for task in available_tasks] if available_tasks else []

            print(f"[OK] PanEcho model loaded successfully with {len(task_names)} tasks")
            print(f"   Total tasks available: {len(task_names)}")
            return model

        except Exception as e:
            print(f"[ERROR] Failed to load PanEcho model: {e}")
            return None

    def load_medsam2(self) -> Optional[Any]:
        """Load MedSAM2 model with real weights."""
        try:
            # Check for local checkpoint first
            checkpoint_path = self.checkpoints_dir / "MedSAM2_US_Heart.pt"
            if checkpoint_path.exists():
                print(f"[OK] Using local MedSAM2 checkpoint: {checkpoint_path}")
                return str(checkpoint_path)

            # Fallback to huggingface
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id="wanglab/MedSAM2", filename="MedSAM2_US_Heart.pt")
            print(f"[OK] Downloaded MedSAM2 model: {model_path}")
            return model_path

        except Exception as e:
            print(f"[ERROR] Failed to load MedSAM2 model: {e}")
            return None

    def load_echoflow(self) -> Optional[Any]:
        """Load EchoFlow model with real weights."""
        try:
            root = Path(__file__).resolve().parents[1]
            candidates = [
                root / "tool_repos" / "EchoFlow",
                root / "tool_repos" / "EchoFlow-main",
                root / "EchoFlow",
            ]
            workspace_root = os.getenv("ECHO_WORKSPACE_ROOT")
            if workspace_root:
                workspace_root = Path(workspace_root)
                candidates.extend(
                    [
                        workspace_root / "tool_repos" / "EchoFlow",
                        workspace_root / "EchoFlow",
                    ]
                )

            echoflow_path = next((path for path in candidates if path.exists()), None)
            if echoflow_path is None:
                return None

            if str(echoflow_path) not in sys.path:
                sys.path.insert(0, str(echoflow_path))

            from echoflow.common.echoflow_model import EchoFlowModel

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = EchoFlowModel(device=device, model_dir=echoflow_path)
            return model if model.load_components() else None

        except Exception:
            import traceback

            traceback.print_exc()
            return None

    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a model by name."""
        if model_name in self._models:
            return self._models[model_name]

        # Load model if not cached
        if model_name == "echo_prime":
            model = self.load_echo_prime()
        elif model_name == "panecho":
            model = self.load_panecho()
        elif model_name == "medsam2":
            model = self.load_medsam2()
        elif model_name == "echoflow":
            model = self.load_echoflow()
        else:
            print(f"[ERROR] Unknown model: {model_name}")
            return None

        if model is not None:
            self._models[model_name] = model

        return model

    def get_available_models(self) -> list:
        """Get list of available models."""
        return ["echo_prime", "panecho", "medsam2", "echoflow"]

    def cleanup(self):
        """Clean up all loaded models."""
        for model_name, model in self._models.items():
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
        self._models.clear()
        print("[OK] All models cleaned up")


# Global model factory
model_factory = ModelFactory()

def get_model(model_name: str) -> Optional[Any]:
    """Get a model using the global factory."""
    return model_factory.get_model(model_name)

def get_available_models() -> list:
    """Get available models using the global factory."""
    return model_factory.get_available_models()

def cleanup_all_models():
    """Clean up all models using the global factory."""
    model_factory.cleanup()
