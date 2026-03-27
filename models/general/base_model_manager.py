"""
Base Model Manager

Provides the base classes for model management in the EchoPilot agent.
"""

import os
import sys
import json
import tempfile
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np


class ModelStatus(Enum):
    """Model status enumeration."""
    NOT_AVAILABLE = "not_available"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    FALLBACK = "fallback"


class ModelConfig:
    """Base configuration class for models."""
    
    def __init__(
        self,
        name: str,
        model_type: str,
        device: Optional[str] = None,
        temp_dir: Optional[str] = None,
        **kwargs
    ):
        self.name = name
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
        # Add any additional configuration parameters
        for key, value in kwargs.items():
            setattr(self, key, value)


class BaseModelManager(ABC):
    """
    Base class for model managers.
    
    This class provides common functionality for managing AI models,
    including initialization, status tracking, and basic operations.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the base model manager.
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.status = ModelStatus.NOT_AVAILABLE
        self.model = None
        self._initialized = False
        
        # Initialize the model
        self._initialize_model()
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the specific model. Must be implemented by subclasses."""
        pass
    
    def _set_status(self, status: ModelStatus):
        """Set the model status."""
        self.status = status
        print(f"Model {self.config.name} status: {status.value}")
    
    def is_ready(self) -> bool:
        """Check if the model is ready for use."""
        return self.status == ModelStatus.READY
    
    def is_available(self) -> bool:
        """Check if the model is available (ready or fallback)."""
        return self.status in [ModelStatus.READY, ModelStatus.FALLBACK]
    
    def get_status(self) -> ModelStatus:
        """Get the current model status."""
        return self.status
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.config.name,
            "type": self.config.model_type,
            "status": self.status.value,
            "device": self.config.device,
            "initialized": self._initialized
        }
    
    @abstractmethod
    def predict(self, input_data: Union[torch.Tensor, List[str], str]) -> Dict[str, Any]:
        """
        Run prediction on input data. Must be implemented by subclasses.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Prediction results dictionary
        """
        pass
    
    def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._set_status(ModelStatus.NOT_AVAILABLE)
        self._initialized = False
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


class MockModelManager(BaseModelManager):
    """
    Mock model manager for testing and fallback purposes.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        if config is None:
            config = ModelConfig(
                name="MockModel",
                model_type="mock",
                device="cpu"
            )
        super().__init__(config)
    
    def _initialize_model(self):
        """Initialize mock model."""
        self._set_status(ModelStatus.READY)
        self._initialized = True
        print("Mock model initialized")
    
    def predict(self, input_data: Union[torch.Tensor, List[str], str]) -> Dict[str, Any]:
        """Mock prediction."""
        return {
            "status": "success",
            "model": "mock",
            "predictions": {
                "mock_prediction": 0.5,
                "confidence": 0.8
            },
            "message": "Mock prediction completed"
        }


class ModelFactory:
    """
    Factory class for creating model managers.
    """
    
    _registered_models = {}
    
    @classmethod
    def register_model(cls, name: str, model_class: type):
        """Register a model class."""
        cls._registered_models[name] = model_class
    
    @classmethod
    def create_model(cls, name: str, config: Optional[ModelConfig] = None) -> BaseModelManager:
        """Create a model instance."""
        if name not in cls._registered_models:
            raise ValueError(f"Unknown model: {name}")
        
        model_class = cls._registered_models[name]
        return model_class(config)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List available models."""
        return list(cls._registered_models.keys())


# Register mock model
ModelFactory.register_model("mock", MockModelManager)
