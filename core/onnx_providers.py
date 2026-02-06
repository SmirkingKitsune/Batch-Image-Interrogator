"""ONNX Runtime Execution Provider settings management."""

import json
import os
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class ProviderPreference(Enum):
    """Execution provider preference options."""
    TENSORRT_CUDA_CPU = "tensorrt_cuda_cpu"
    CUDA_CPU = "cuda_cpu"
    CPU_ONLY = "cpu_only"

    @classmethod
    def from_string(cls, value: str) -> 'ProviderPreference':
        """Convert string to enum, with fallback to CUDA_CPU."""
        for member in cls:
            if member.value == value:
                return member
        return cls.CUDA_CPU

    def display_name(self) -> str:
        """Get human-readable display name."""
        names = {
            ProviderPreference.TENSORRT_CUDA_CPU: "TensorRT > CUDA > CPU",
            ProviderPreference.CUDA_CPU: "CUDA > CPU (Default)",
            ProviderPreference.CPU_ONLY: "CPU Only"
        }
        return names.get(self, self.value)

    def description(self) -> str:
        """Get description of what this preference does."""
        descriptions = {
            ProviderPreference.TENSORRT_CUDA_CPU: (
                "Use TensorRT for fastest inference (requires TensorRT installed). "
                "Falls back to CUDA, then CPU if unavailable."
            ),
            ProviderPreference.CUDA_CPU: (
                "Use CUDA GPU acceleration. Falls back to CPU if unavailable. "
                "Good balance of speed and compatibility."
            ),
            ProviderPreference.CPU_ONLY: (
                "Force CPU-only inference. Use this if you encounter GPU issues "
                "or want to free up GPU memory."
            )
        }
        return descriptions.get(self, "")


class ONNXProviderSettings:
    """Manages ONNX Runtime execution provider settings."""

    # Default TensorRT options for optimal performance
    DEFAULT_TRT_OPTIONS = {
        'trt_max_workspace_size': 2147483648,  # 2GB
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': str(Path.home() / '.cache' / 'tensorrt_engines')
    }

    def __init__(self, settings_file: str = "onnx_provider_settings.json"):
        """
        Initialize ONNX Provider Settings.

        Args:
            settings_file: Path to JSON settings file
        """
        self.settings_file = Path(settings_file)
        self.preference = ProviderPreference.CUDA_CPU
        self.trt_options = self.DEFAULT_TRT_OPTIONS.copy()

        # Cache for available providers
        self._available_providers: Optional[List[str]] = None

        # Load saved settings
        self.load_settings()

        # Ensure TensorRT cache directory exists
        cache_path = Path(self.trt_options['trt_engine_cache_path'])
        cache_path.mkdir(parents=True, exist_ok=True)

    def load_settings(self):
        """Load settings from file."""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.preference = ProviderPreference.from_string(
                        data.get('preference', 'cuda_cpu')
                    )
                    # Merge saved TRT options with defaults
                    saved_trt = data.get('trt_options', {})
                    self.trt_options.update(saved_trt)
            except Exception as e:
                print(f"Error loading ONNX provider settings: {e}")

    def save_settings(self):
        """Save settings to file."""
        try:
            data = {
                'preference': self.preference.value,
                'trt_options': self.trt_options
            }
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving ONNX provider settings: {e}")

    def get_available_providers(self, force_refresh: bool = False) -> List[str]:
        """
        Get list of available ONNX Runtime execution providers.

        Args:
            force_refresh: Force re-detection of providers

        Returns:
            List of available provider names
        """
        if self._available_providers is None or force_refresh:
            try:
                import onnxruntime as ort
                self._available_providers = ort.get_available_providers()
            except ImportError:
                self._available_providers = ['CPUExecutionProvider']
            except Exception as e:
                print(f"Error detecting ONNX providers: {e}")
                self._available_providers = ['CPUExecutionProvider']

        return self._available_providers

    def is_tensorrt_available(self) -> bool:
        """Check if TensorRT execution provider is available."""
        return 'TensorrtExecutionProvider' in self.get_available_providers()

    def is_cuda_available(self) -> bool:
        """Check if CUDA execution provider is available."""
        return 'CUDAExecutionProvider' in self.get_available_providers()

    def set_preference(self, preference: ProviderPreference):
        """Set the provider preference and save."""
        self.preference = preference
        self.save_settings()

    def get_provider_chain(self, device: str = 'cuda') -> List[str]:
        """
        Get ordered list of providers based on preference and availability.

        Args:
            device: Requested device ('cuda' or 'cpu')

        Returns:
            List of provider names to try in order
        """
        available = self.get_available_providers()

        # CPU-only mode or CPU device requested
        if self.preference == ProviderPreference.CPU_ONLY or device == 'cpu':
            return ['CPUExecutionProvider']

        providers = []

        # TensorRT preference
        if self.preference == ProviderPreference.TENSORRT_CUDA_CPU:
            if 'TensorrtExecutionProvider' in available:
                providers.append('TensorrtExecutionProvider')
            if 'CUDAExecutionProvider' in available:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')

        # CUDA preference (default)
        elif self.preference == ProviderPreference.CUDA_CPU:
            if 'CUDAExecutionProvider' in available:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')

        return providers if providers else ['CPUExecutionProvider']

    def get_provider_options(self) -> List[Dict]:
        """
        Get provider options for each provider in the chain.

        Returns:
            List of option dicts matching provider chain order
        """
        chain = self.get_provider_chain()
        options = []

        for provider in chain:
            if provider == 'TensorrtExecutionProvider':
                options.append(self.trt_options.copy())
            elif provider == 'CUDAExecutionProvider':
                # Basic CUDA options
                options.append({
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit
                    'cudnn_conv_algo_search': 'EXHAUSTIVE'
                })
            else:
                options.append({})

        return options

    def get_session_options(self):
        """
        Get ONNX Runtime SessionOptions configured for optimal performance.

        Returns:
            onnxruntime.SessionOptions instance
        """
        try:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()

            # Enable graph optimizations (important for TensorRT)
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Enable memory pattern optimization
            sess_options.enable_mem_pattern = True
            sess_options.enable_mem_reuse = True

            # Set execution mode for better parallelism
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

            # Set inter/intra op parallelism
            sess_options.inter_op_num_threads = 0  # Use default
            sess_options.intra_op_num_threads = 0  # Use default

            return sess_options

        except ImportError:
            return None
        except Exception as e:
            print(f"Error creating session options: {e}")
            return None

    def create_inference_session(self, model_path: str, device: str = 'cuda'):
        """
        Create an ONNX Runtime InferenceSession with appropriate providers.

        Args:
            model_path: Path to the ONNX model file
            device: Requested device ('cuda' or 'cpu')

        Returns:
            onnxruntime.InferenceSession instance
        """
        import onnxruntime as ort

        providers = self.get_provider_chain(device)
        provider_options = self.get_provider_options()[:len(providers)]
        sess_options = self.get_session_options()

        # Create session with providers and options
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options if provider_options else None
        )

        return session

    def get_active_provider(self, session) -> str:
        """
        Get the provider that is actually being used by a session.

        Args:
            session: ONNX Runtime InferenceSession

        Returns:
            Name of the active provider
        """
        try:
            providers = session.get_providers()
            return providers[0] if providers else 'Unknown'
        except Exception:
            return 'Unknown'

    def get_status_info(self) -> Dict:
        """
        Get status information about providers for UI display.

        Returns:
            Dict with provider availability and current preference
        """
        available = self.get_available_providers()

        return {
            'tensorrt_available': 'TensorrtExecutionProvider' in available,
            'cuda_available': 'CUDAExecutionProvider' in available,
            'cpu_available': 'CPUExecutionProvider' in available,
            'all_providers': available,
            'current_preference': self.preference,
            'preference_display': self.preference.display_name(),
            'preference_description': self.preference.description()
        }

    def get_statistics(self) -> Dict:
        """Get settings statistics for display."""
        status = self.get_status_info()
        return {
            'preference': status['preference_display'],
            'tensorrt': 'Available' if status['tensorrt_available'] else 'Not available',
            'cuda': 'Available' if status['cuda_available'] else 'Not available',
            'providers_count': len(status['all_providers'])
        }
