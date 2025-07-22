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
        
        # Start monitoring task
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start background monitoring tasks."""
        async def monitor_resources():
            while True:
                try:
                    await self._check_memory_usage()
                    await self._cleanup_idle_models()
                    await asyncio.sleep(self.cleanup_interval)
                except Exception as e:
                    self.logger.error(f"Error in resource monitoring: {e}")
        
        asyncio.create_task(monitor_resources())
    
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
        # Unload all non-critical models
        for model_id, resource in list(self._models.items()):
            if resource.priority != ModelPriority.CRITICAL:
                await self.unload_model(model_id)
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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
        Load a model with resource management.
        
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
            return resource.model, resource.device
        
        # Select device
        device = self._select_device(preferred_device)
        
        # Ensure memory available
        await self._ensure_memory_available(model_type, device)
        
        # Load the model
        try:
            model = await loader_func(device)
            
            # Track memory usage
            memory_before = self._get_memory_usage(device)
            memory_after = self._get_memory_usage(device)
            memory_used = memory_after - memory_before
            
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
            
            return model, device
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    async def unload_model(self, model_id: str):
        """
        Unload a model and free its resources.
        
        Args:
            model_id: ID of model to unload
        """
        if model_id not in self._models:
            return
        
        resource = self._models[model_id]
        
        try:
            # Clear from device
            if hasattr(resource.model, 'cpu'):
                resource.model.cpu()
            
            # Remove from device tracking
            if resource.device in self._device_allocations:
                self._device_allocations[resource.device].remove(model_id)
            
            # Clear model reference
            del self._models[model_id]
            
            # Force cleanup
            if torch.cuda.is_available() and resource.device.startswith('cuda'):
                torch.cuda.empty_cache()
                
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
    
    async def _ensure_memory_available(self, model_type: ModelType, device: str):
        """Ensure sufficient memory is available for a new model."""
        if device == "cpu":
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_high:
                await self._selective_cleanup()
            if memory.percent > self.memory_critical:
                await self._emergency_cleanup()
                
        elif device.startswith("cuda"):
            try:
                device_idx = int(device.split(":")[-1])
                total = torch.cuda.get_device_properties(device_idx).total_memory
                used = torch.cuda.memory_allocated(device_idx)
                if used / total > 0.9:  # 90% GPU memory used
                    await self._selective_cleanup()
            except:
                pass
    
    def get_model_info(self, model_id: str) -> Optional[ModelResource]:
        """Get information about a loaded model."""
        return self._models.get(model_id)
