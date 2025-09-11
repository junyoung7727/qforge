"""
Efficient Device Management for CUDA GPU Training
Centralized device handling with automatic optimization and memory management
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Union, Optional, List
import warnings
from contextlib import contextmanager
import gc


class DeviceManager:
    """
    Centralized device management for efficient CUDA training
    Handles device detection, tensor movement, and memory optimization
    """
    
    def __init__(self, preferred_device: Optional[str] = None, enable_amp: bool = True):
        """
        Initialize device manager
        
        Args:
            preferred_device: 'cuda', 'cpu', or None for auto-detection
            enable_amp: Enable automatic mixed precision for CUDA
        """
        self.device = self._detect_optimal_device(preferred_device)
        self.enable_amp = enable_amp and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.enable_amp else None
        
        # Device info
        self._log_device_info()
        
        # Memory management
        self._setup_memory_optimization()
    
    def _detect_optimal_device(self, preferred: Optional[str]) -> torch.device:
        """Detect and return optimal device"""
        if preferred:
            if preferred == 'cuda' and not torch.cuda.is_available():
                warnings.warn("CUDA requested but not available, falling back to CPU")
                return torch.device('cpu')
            return torch.device(preferred)
        
        # Auto-detection
        if torch.cuda.is_available():
            # Select GPU with most free memory
            if torch.cuda.device_count() > 1:
                free_memory = []
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    free_mem = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                    free_memory.append(free_mem)
                best_gpu = free_memory.index(max(free_memory))
                return torch.device(f'cuda:{best_gpu}')
            return torch.device('cuda:0')
        
        return torch.device('cpu')
    
    def _log_device_info(self):
        """Log device information"""
        print(f"🔧 Device Manager Initialized:")
        print(f"   Device: {self.device}")
        
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(self.device)
            print(f"   GPU: {props.name}")
            print(f"   Memory: {props.total_memory / 1e9:.1f}GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            print(f"   AMP Enabled: {self.enable_amp}")
        else:
            print(f"   CPU Cores: {torch.get_num_threads()}")
    
    def _setup_memory_optimization(self):
        """Setup memory optimization for CUDA"""
        if self.device.type == 'cuda':
            # Enable memory pool for faster allocation
            torch.cuda.empty_cache()
            
            # Set memory fraction if needed (optional)
            # torch.cuda.set_per_process_memory_fraction(0.9)
            
            # Enable cudnn benchmarking for consistent input sizes
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def to_device(self, data: Union[torch.Tensor, Dict, List, Any], non_blocking: bool = True) -> Any:
        """
        Efficiently move data to device
        
        Args:
            data: Data to move (tensor, dict, list, or any nested structure)
            non_blocking: Use non-blocking transfer for CUDA
            
        Returns:
            Data moved to device
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=non_blocking and self.device.type == 'cuda')
        
        elif isinstance(data, dict):
            return {k: self.to_device(v, non_blocking) for k, v in data.items()}
        
        elif isinstance(data, (list, tuple)):
            moved_data = [self.to_device(item, non_blocking) for item in data]
            return type(data)(moved_data)
        
        else:
            # Non-tensor data (strings, numbers, etc.)
            return data
    
    def move_model_to_device(self, model: nn.Module) -> nn.Module:
        """Move model to device with optimization"""
        model = model.to(self.device)
        
        if self.device.type == 'cuda':
            # Enable gradient checkpointing for large models
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
        
        return model
    
    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision"""
        if self.enable_amp:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
    
    def scale_loss_and_backward(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """Handle loss scaling and backward pass"""
        if self.enable_amp and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def optimizer_step(self, optimizer: torch.optim.Optimizer):
        """Handle optimizer step with scaling"""
        if self.enable_amp and self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': total - reserved,
                'utilization': reserved / total * 100
            }
        else:
            return {'device': 'cpu', 'info': 'Memory info not available for CPU'}
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
    
    def synchronize(self):
        """Synchronize device operations"""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
    
    def set_deterministic(self, seed: int = 42):
        """Set deterministic behavior for reproducibility"""
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Note: This may reduce performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def __str__(self) -> str:
        return f"DeviceManager(device={self.device}, amp={self.enable_amp})"


# Global device manager instance
_global_device_manager: Optional[DeviceManager] = None


def get_device_manager(preferred_device: Optional[str] = None, enable_amp: bool = True) -> DeviceManager:
    """Get or create global device manager instance"""
    global _global_device_manager
    
    if _global_device_manager is None:
        _global_device_manager = DeviceManager(preferred_device, enable_amp)
    
    return _global_device_manager


def reset_device_manager():
    """Reset global device manager (useful for testing)"""
    global _global_device_manager
    _global_device_manager = None


# Convenience functions
def to_device(data: Any, non_blocking: bool = True) -> Any:
    """Convenience function to move data to device"""
    return get_device_manager().to_device(data, non_blocking)


def get_device() -> torch.device:
    """Get current device"""
    return get_device_manager().device


def cleanup_memory():
    """Cleanup GPU memory"""
    get_device_manager().cleanup_memory()


def get_memory_info() -> Dict[str, float]:
    """Get memory information"""
    return get_device_manager().get_memory_info()


if __name__ == "__main__":
    # Test device manager
    dm = DeviceManager()
    print(f"Device Manager: {dm}")
    print(f"Memory Info: {dm.get_memory_info()}")
    
    # Test tensor movement
    x = torch.randn(10, 10)
    x_device = dm.to_device(x)
    print(f"Tensor moved to: {x_device.device}")
    
    # Test dict movement
    data = {
        'tensor1': torch.randn(5, 5),
        'tensor2': torch.randn(3, 3),
        'metadata': {'info': 'test'}
    }
    data_device = dm.to_device(data)
    print(f"Dict tensors moved to: {data_device['tensor1'].device}")
