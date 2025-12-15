"""Abstract base class for image interrogators."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class BaseInterrogator(ABC):
    """Abstract base class for image interrogators."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_loaded = False
        self.config = {}
    
    @abstractmethod
    def load_model(self, **kwargs):
        """Load the interrogation model."""
        pass
    
    @abstractmethod
    def interrogate(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """
        Interrogate an image and return results.
        
        Returns:
            Dict containing:
                - tags: List[str]
                - confidence_scores: Optional[Dict[str, float]]
                - raw_output: Optional[str]
        """
        pass
    
    @abstractmethod
    def get_model_type(self) -> str:
        """Return the model type identifier."""
        pass
    
    def unload_model(self):
        """Unload model from memory."""
        self.model = None
        self.is_loaded = False
    
    def get_config(self) -> Dict:
        """Get current configuration."""
        return self.config.copy()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup."""
        self.unload_model()
