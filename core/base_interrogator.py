"""Abstract base class for image interrogators."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class BaseInterrogator(ABC):
    """Abstract base class for image interrogators."""

    # Bump when a fix changes what interrogate() returns for the same image and
    # settings. Cached rows are stamped with the version that produced them, and
    # rows from an older version are regenerated instead of reused.
    RESULT_VERSION = 0

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

    def result_signature(self) -> Optional[Dict[str, Any]]:
        """
        Settings that shape interrogate()'s output, saved with each result
        so reuse_cached_result() can tell whether a cached one still applies.
        """
        return None

    def reuse_cached_result(self, cached: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Return a cached result as the current settings would produce it, or
        None to run the model again. `cached['config']` holds the signature
        it was saved with, if any.

        By default a result is reused only when it was saved with the current
        signature; interrogators without one reuse whatever is cached.
        """
        signature = self.result_signature()
        if signature is None or cached.get('config') == signature:
            return cached
        return None

    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup."""
        self.unload_model()
