"""
Model Resource Management System

Provides intelligent model loading, unloading, and memory management
for OCR engines and related models.
"""

import time
import asyncio
import gc
from typing import Dict, Any, Optional, Set, Tuple
import psutil
import torch
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger


class ModelType(Enum):
    """Types of models managed by the system."""
    OCR = "ocr"
    STRUCTURE = "structure"
    RECOGNITION = "recognition"
    EXTRACTION = "extraction"
    AUXILIARY = "auxiliary"


class ModelPriority(Enum):
    """Priority levels for model loading/unloading decisions."""
    CRITICAL = 3  # Never unload
    HIGH = 2      # Unload only under severe memory pressure
    MEDIUM = 1    # Normal unloading candidate
    LOW = 0       # Unload first when needed


@dataclass
class ModelResource:
    """Information about a loaded model."""
    model_type: ModelType
    priority: ModelPriority
    last_used: float
    memory_usage: int  # Bytes
    device: str
    model: Any
    metadata: Dict[str, Any]


class ResourceManager:
    """
    Manages model loading, unloading, and memory allocation.
    
    Features:
    1. Intelligent model loading/unloading
    2. Memory usage monitoring
    3. Device allocation optimization
    4. Automatic cleanup of unused models
    5. Priority-based resource management
    """
    
    def __init__(self, config: ConfigManager):
        """Initialize the resource manager."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Load configuration
        self.resource_config = config.get("resources", {})
        
        # Memory thresholds (percentage of system memory)
        self.memory_high = self.resource_config.get("memory_high_threshold", 80)
        self.memory_critical = self.resource_config.get("memory_critical_threshold", 90)
        
        # Model cleanup settings
        self.idle_threshold = self.resource_config.get("model_idle_threshold", 300)  # 5 minutes
        self.cleanup_interval = self.resource_config.get("cleanup_interval", 60)  # 1 minute
        
        # Track loaded models
        self._models: Dict[str, ModelResource] = {}
        self._device_allocations: Dict[str, Set[str]] = {}
        
        # Memory tracking for leak detection
        self._memory_history = []
        self._memory_check_interval = 60  # 1 minute
        self._max_memory_history = 60  # Keep 1 hour of history
        
        # Initialize memory tracking
        self._track_initial_memory()
        
        # Start monitoring task
        self._start_monitoring()
    
    def _track_initial_memory(self):
        """Track initial memory state for leak detection."""
        try:
            cpu_memory = psutil.Process().memory_info().rss
            gpu_memory = None
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated()
            
            self._memory_history.append({
                'timestamp': time.time(),
                'cpu_memory': cpu_memory,
                'gpu_memory': gpu_memory
            })
            
        except Exception as e:
            self.logger.warning(f"Failed to track initial memory state: {e}")
    
    def _check_for_memory_leaks(self):
        """Check for potential memory leaks based on history."""
        if len(self._memory_history) < 2:
            return
        
        try:
            # Calculate memory growth rate
            current = self._memory_history[-1]
            oldest = self._memory_history[0]
            time_diff = current['timestamp'] - oldest['timestamp']
            
            if time_diff < 60:  # Need at least 1 minute of history
                return
            
            # Check CPU memory growth
            if current['cpu_memory'] and oldest['cpu_memory']:
                cpu_growth = current['cpu_memory'] - oldest['cpu_memory']
                cpu_growth_rate = cpu_growth / time_diff
                
                if cpu_growth_rate > 1024 * 1024 * 10:  # More than 10MB/s growth
                    self.logger.warning(
                        f"Potential CPU memory leak detected! "
                        f"Growth rate: {cpu_growth_rate / 1024 / 1024:.2f} MB/s"
                    )
            
            # Check GPU memory growth
            if (current['gpu_memory'] is not None and 
                oldest['gpu_memory'] is not None):
                gpu_growth = current['gpu_memory'] - oldest['gpu_memory']
                gpu_growth_rate = gpu_growth / time_diff
                
                if gpu_growth_rate > 1024 * 1024 * 10:  # More than 10MB/s growth
                    self.logger.warning(
                        f"Potential GPU memory leak detected! "
                        f"Growth rate: {gpu_growth_rate / 1024 / 1024:.2f} MB/s"
                    )
            
        except Exception as e:
            self.logger.error(f"Error checking for memory leaks: {e}")
    
    def _start_monitoring(self):
        """Start background monitoring tasks."""
        async def monitor_resources():
            last_memory_check = 0
            
            while True:
                try:
                    # Regular resource checks
                    await self._check_memory_usage()
                    await self._cleanup_idle_models()
                    
                    # Memory tracking for leak detection
                    current_time = time.time()
                    if current_time - last_memory_check >= self._memory_check_interval:
                        self._track_initial_memory()  # Add current memory state to history
                        self._check_for_memory_leaks()  # Check for potential leaks
                        
                        # Trim memory history
                        if len(self._memory_history) > self._max_memory_history:
                            self._memory_history = self._memory_history[-self._max_memory_history:]
                        
                        last_memory_check = current_time
                    
                    await asyncio.sleep(self.cleanup_interval)
                except Exception as e:
                    self.logger.error(f"Error in resource monitoring: {e}", exc_info=True)
        
        try:
            # Try to create the task if there's a running event loop
            loop = asyncio.get_running_loop()
            loop.create_task(monitor_resources())
        except RuntimeError:
            # No running event loop, skip monitoring for now
            self.logger.warning("No running event loop, skipping resource monitoring")
            pass
    
    async def _check_memory_usage(self):
        """Monitor system memory usage and take action if needed."""
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        if memory_percent > self.memory_critical:
            self.logger.warning(f"Critical memory usage: {memory_percent}%")
            await self._emergency_cleanup()
        elif memory_percent > self.memory_high:
            self.logger.info(f"High memory usage: {memory_percent}%")
            await self._selective_cleanup()
    
    async def _cleanup_idle_models(self):
        """Clean up models that have been idle too long."""
        current_time = time.time()
        
        for model_id, resource in list(self._models.items()):
            if (
                resource.priority != ModelPriority.CRITICAL and
                current_time - resource.last_used > self.idle_threshold
            ):
                await self.unload_model(model_id)
    
    async def _emergency_cleanup(self):
        """Aggressive cleanup during critical memory pressure."""
        self.logger.warning("Performing emergency cleanup of model resources")
        try:
            # Create a copy of models to avoid modification during iteration
            models_to_unload = [(model_id, resource) for model_id, resource 
                              in self._models.items() 
                              if resource.priority != ModelPriority.CRITICAL]
            
            # Unload all non-critical models
            for model_id, _ in models_to_unload:
                try:
                    await self.unload_model(model_id)
                except Exception as e:
                    self.logger.error(f"Error unloading model {model_id} during emergency cleanup: {e}")
            
            # Force garbage collection
            try:
                gc.collect()
            except Exception as e:
                self.logger.error(f"Error during garbage collection: {e}")
                
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Ensure all CUDA operations are complete
                except Exception as e:
                    self.logger.error(f"Error clearing CUDA cache: {e}")
                    
            # Check if cleanup was successful
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_critical:
                self.logger.error(f"Emergency cleanup failed to reduce memory usage: {memory.percent}%")
            else:
                self.logger.info(f"Emergency cleanup successful. Memory usage: {memory.percent}%")
                
        except Exception as e:
            self.logger.error(f"Emergency cleanup failed: {e}", exc_info=True)
    
    async def _selective_cleanup(self):
        """Selectively unload models based on priority and idle time."""
        current_time = time.time()
        
        # Calculate scores for each model
        scores = []
        for model_id, resource in self._models.items():
            if resource.priority == ModelPriority.CRITICAL:
                continue
                
            idle_time = current_time - resource.last_used
            priority_score = resource.priority.value
            memory_score = resource.memory_usage / (1024 * 1024)  # Convert to MB
            
            score = (
                idle_time * 0.5 +
                (3 - priority_score) * 0.3 +
                memory_score * 0.2
            )
            scores.append((model_id, score))
        
        # Unload models with highest scores until memory usage improves
        for model_id, _ in sorted(scores, key=lambda x: x[1], reverse=True):
            memory = psutil.virtual_memory()
            if memory.percent <= self.memory_high:
                break
                
            await self.unload_model(model_id)
    
    async def load_model(
        self,
        model_id: str,
        model_type: ModelType,
        priority: ModelPriority,
        loader_func: callable,
        preferred_device: str = None,
        metadata: Dict[str, Any] = None
    ) -> Tuple[Any, str]:
        """
        Load a model with resource management and sequential loading.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model being loaded
            priority: Loading/unloading priority
            loader_func: Function to load the model
            preferred_device: Preferred device for model
            metadata: Additional model information
            
        Returns:
            Tuple of (loaded_model, device)
        """
        # Check if already loaded
        if model_id in self._models:
            resource = self._models[model_id]
            resource.last_used = time.time()
            self.logger.info(f"Model {model_id} already loaded, returning existing instance")
            return resource.model, resource.device
        
        # Select device
        device = self._select_device(preferred_device)
        self.logger.info(f"Loading model {model_id} on device {device}")
        
        # Ensure memory available with proper cleanup
        await self._ensure_memory_available(model_type, device)
        
        # Load the model with comprehensive error handling
        try:
            self.logger.info(f"Starting model loading for {model_id}...")
            model = await loader_func(device)
            
            # Track memory usage more accurately
            memory_before = self._get_memory_usage(device)
            memory_after = self._get_memory_usage(device)
            memory_used = max(0, memory_after - memory_before)
            
            # Register model
            self._models[model_id] = ModelResource(
                model_type=model_type,
                priority=priority,
                last_used=time.time(),
                memory_usage=memory_used,
                device=device,
                model=model,
                metadata=metadata or {}
            )
            
            # Track device allocation
            if device not in self._device_allocations:
                self._device_allocations[device] = set()
            self._device_allocations[device].add(model_id)
            
            self.logger.info(f"Successfully loaded model {model_id} on {device} (memory: {memory_used / 1024**3:.2f} GB)")
            return model, device
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            # Clean up any partial loading
            if model_id in self._models:
                del self._models[model_id]
            raise RuntimeError(f"Model loading failed for {model_id}: {e}")

    async def unload_model(self, model_id: str):
        """
        Unload a model and free its resources with proper cleanup.
        
        Args:
            model_id: ID of model to unload
        """
        if model_id not in self._models:
            self.logger.debug(f"Model {model_id} not found for unloading")
            return
        
        resource = self._models[model_id]
        self.logger.info(f"Unloading model {model_id} from {resource.device}")
        
        try:
            # Clear from device
            if resource.model is not None:
                try:
                    # Move to CPU if possible
                    if hasattr(resource.model, 'cpu'):
                        resource.model.cpu()
                    
                    # Clear CUDA memory if using GPU
                    if hasattr(resource.model, 'to'):
                        resource.model.to('cpu')
                    
                    # Delete model reference
                    del resource.model
                except Exception as e:
                    self.logger.warning(f"Error moving model to CPU: {e}")
            
            # Remove from device tracking
            if resource.device in self._device_allocations:
                try:
                    self._device_allocations[resource.device].discard(model_id)
                except Exception as e:
                    self.logger.warning(f"Error updating device allocations: {e}")
            
            # Clear model reference from registry
            try:
                del self._models[model_id]
            except Exception as e:
                self.logger.warning(f"Error removing model from registry: {e}")
            
            # Force cleanup with proper error handling
            try:
                if torch.cuda.is_available() and resource.device.startswith('cuda'):
                    device_id = int(resource.device.split(':')[1])
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(device_id)
            except Exception as e:
                self.logger.warning(f"Error clearing CUDA cache: {e}")
            
            try:
                gc.collect()
            except Exception as e:
                self.logger.warning(f"Error during garbage collection: {e}")
            
            self.logger.info(f"Successfully unloaded model {model_id}")
            
        except Exception as e:
            self.logger.error(f"Error during model unloading: {e}", exc_info=True)
            # Even if unloading fails, try to remove from registry
            try:
                del self._models[model_id]
            except:
                pass
                
        except Exception as e:
            self.logger.error(f"Error unloading model {model_id}: {e}")

    def _select_device(self, preferred_device: Optional[str] = None) -> str:
        """Select the best device for a new model."""
        if preferred_device and self._is_device_available(preferred_device):
            return preferred_device
            
        if torch.cuda.is_available():
            # Select GPU with most free memory
            free_memory = []
            for i in range(torch.cuda.device_count()):
                free_memory.append(
                    (i, torch.cuda.get_device_properties(i).total_memory -
                        torch.cuda.memory_allocated(i))
                )
            if free_memory:
                best_gpu = max(free_memory, key=lambda x: x[1])[0]
                return f"cuda:{best_gpu}"
        
        return "cpu"
    
    def _is_device_available(self, device: str) -> bool:
        """Check if a device is available for use."""
        if device == "cpu":
            return True
            
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                return False
            try:
                device_idx = int(device.split(":")[-1])
                return device_idx < torch.cuda.device_count()
            except:
                return False
        
        return False
    
    def _get_memory_usage(self, device: str) -> int:
        """Get current memory usage for a device."""
        if device == "cpu":
            return psutil.Process().memory_info().rss
            
        if device.startswith("cuda"):
            try:
                device_idx = int(device.split(":")[-1])
                return torch.cuda.memory_allocated(device_idx)
            except:
                return 0
        
        return 0
    
    async def _ensure_memory_available(self, model_type, device: str):
        """Ensure sufficient memory is available for model loading."""
        try:
            # Get current memory usage
            current_memory = self._get_memory_usage(device)
            total_memory = self._get_total_memory(device)
            memory_usage_percent = (current_memory / total_memory) * 100
            
            self.logger.debug(f"Current memory usage on {device}: {memory_usage_percent:.1f}%")
            
            # If memory usage is high, perform cleanup
            if memory_usage_percent > self.memory_high:
                self.logger.warning(f"High memory usage on {device}: {memory_usage_percent:.1f}%")
                await self._selective_cleanup()
                
                # Check again after cleanup
                current_memory = self._get_memory_usage(device)
                memory_usage_percent = (current_memory / total_memory) * 100
                
                if memory_usage_percent > self.memory_critical:
                    self.logger.error(f"Critical memory usage on {device}: {memory_usage_percent:.1f}%")
                    await self._emergency_cleanup()
                    
        except Exception as e:
            self.logger.error(f"Error ensuring memory availability: {e}")

    def _get_total_memory(self, device: str) -> int:
        """Get total memory available on device."""
        try:
            if device.startswith('cuda') and torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory
            else:
                # For CPU, use system memory
                import psutil
                return psutil.virtual_memory().total
        except Exception as e:
            self.logger.warning(f"Could not get total memory for {device}: {e}")
            return 8 * 1024**3  # Default to 8GB
    
    def get_model_info(self, model_id: str) -> Optional[ModelResource]:
        """Get information about a loaded model."""
        return self._models.get(model_id)
