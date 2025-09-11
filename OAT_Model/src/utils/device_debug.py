"""
Device Debugging Utilities
디바이스 미스매치 에러를 정확하게 디버깅하기 위한 유틸리티들
"""

import torch
import traceback
import functools
from typing import Dict, Any, Optional, Union, List
from contextlib import contextmanager


class DeviceTracker:
    """디바이스 상태를 추적하는 클래스"""
    
    def __init__(self):
        self.device_log = []
        self.enabled = True
    
    def log_tensor_device(self, name: str, tensor: torch.Tensor, location: str = ""):
        """텐서의 디바이스 정보를 로깅"""
        if not self.enabled:
            return
            
        device_info = {
            'name': name,
            'device': str(tensor.device),
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype),
            'location': location,
            'stack_trace': traceback.format_stack()[-3:-1]  # 호출 위치 추적
        }
        self.device_log.append(device_info)
        print(f"🔍 DEVICE DEBUG: {name} -> {tensor.device} {tuple(tensor.shape)} at {location}")
    
    def log_model_device(self, model: torch.nn.Module, name: str = "model"):
        """모델의 디바이스 정보를 로깅"""
        if not self.enabled:
            return
            
        try:
            model_device = next(model.parameters()).device
            print(f"🏗️ MODEL DEVICE: {name} -> {model_device}")
            return model_device
        except StopIteration:
            print(f"⚠️ MODEL DEVICE: {name} -> No parameters found")
            return None
    
    def clear_log(self):
        """로그 초기화"""
        self.device_log.clear()
    
    def print_summary(self):
        """디바이스 로그 요약 출력"""
        if not self.device_log:
            print("📋 No device operations logged")
            return
            
        print("\n📋 DEVICE OPERATION SUMMARY:")
        print("=" * 60)
        
        devices = set()
        for entry in self.device_log:
            devices.add(entry['device'])
            print(f"  {entry['name']:20} | {entry['device']:10} | {entry['location']}")
        
        print(f"\n🎯 Devices used: {', '.join(devices)}")
        if len(devices) > 1:
            print("⚠️ WARNING: Multiple devices detected - potential mismatch!")


# 전역 디바이스 트래커
device_tracker = DeviceTracker()


def debug_tensor_device(tensor: torch.Tensor, name: str, location: str = "") -> torch.Tensor:
    """텐서의 디바이스를 디버깅하고 반환"""
    device_tracker.log_tensor_device(name, tensor, location)
    return tensor


def debug_model_device(model: torch.nn.Module, name: str = "model") -> Optional[torch.device]:
    """모델의 디바이스를 디버깅하고 반환"""
    return device_tracker.log_model_device(model, name)


@contextmanager
def device_debug_context(name: str = "operation"):
    """디바이스 디버깅 컨텍스트 매니저"""
    print(f"\n🚀 Starting device debug context: {name}")
    device_tracker.clear_log()
    
    try:
        yield device_tracker
    except RuntimeError as e:
        if "device" in str(e).lower():
            print(f"\n💥 DEVICE ERROR in {name}: {e}")
            device_tracker.print_summary()
            print("\n🔍 Error occurred at:")
            traceback.print_exc()
        raise
    finally:
        print(f"\n✅ Ending device debug context: {name}")
        device_tracker.print_summary()


def validate_tensor_devices(*tensors, expected_device: Optional[torch.device] = None, 
                          names: Optional[List[str]] = None) -> bool:
    """여러 텐서의 디바이스가 일치하는지 검증"""
    if not tensors:
        return True
    
    if names is None:
        names = [f"tensor_{i}" for i in range(len(tensors))]
    
    devices = []
    for i, tensor in enumerate(tensors):
        if isinstance(tensor, torch.Tensor):
            devices.append(tensor.device)
            device_tracker.log_tensor_device(names[i], tensor, "validation")
        else:
            print(f"⚠️ {names[i]} is not a tensor: {type(tensor)}")
    
    # 모든 디바이스가 같은지 확인
    if len(set(str(d) for d in devices)) > 1:
        print(f"❌ DEVICE MISMATCH detected!")
        for name, device in zip(names, devices):
            print(f"  {name}: {device}")
        return False
    
    # 예상 디바이스와 일치하는지 확인
    if expected_device is not None and devices:
        if str(devices[0]) != str(expected_device):
            print(f"❌ DEVICE MISMATCH: Expected {expected_device}, got {devices[0]}")
            return False
    
    print(f"✅ All tensors on same device: {devices[0] if devices else 'None'}")
    return True


def device_safe_operation(func):
    """디바이스 안전 연산을 위한 데코레이터"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        
        # 입력 텐서들의 디바이스 로깅
        tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]
        tensor_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
        
        print(f"\n🔧 Calling {func_name}")
        for i, tensor in enumerate(tensor_args):
            device_tracker.log_tensor_device(f"arg_{i}", tensor, func_name)
        
        for name, tensor in tensor_kwargs.items():
            device_tracker.log_tensor_device(f"kwarg_{name}", tensor, func_name)
        
        try:
            result = func(*args, **kwargs)
            
            # 결과 텐서의 디바이스 로깅
            if isinstance(result, torch.Tensor):
                device_tracker.log_tensor_device("result", result, func_name)
            elif isinstance(result, (list, tuple)):
                for i, item in enumerate(result):
                    if isinstance(item, torch.Tensor):
                        device_tracker.log_tensor_device(f"result_{i}", item, func_name)
            elif isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, torch.Tensor):
                        device_tracker.log_tensor_device(f"result_{key}", value, func_name)
            
            return result
            
        except RuntimeError as e:
            if "device" in str(e).lower():
                print(f"\n💥 DEVICE ERROR in {func_name}: {e}")
                device_tracker.print_summary()
            raise
    
    return wrapper


def enable_device_debugging():
    """디바이스 디버깅 활성화"""
    device_tracker.enabled = True
    print("🔍 Device debugging enabled")


def disable_device_debugging():
    """디바이스 디버깅 비활성화"""
    device_tracker.enabled = False
    print("🔇 Device debugging disabled")


# 편의 함수들
def check_device(tensor: torch.Tensor, name: str = "tensor") -> str:
    """텐서의 디바이스를 확인하고 출력"""
    device = str(tensor.device)
    print(f"📍 {name}: {device} {tuple(tensor.shape)}")
    return device


def move_to_device(tensor: torch.Tensor, device: torch.device, name: str = "tensor") -> torch.Tensor:
    """텐서를 디바이스로 이동하며 디버깅"""
    old_device = tensor.device
    new_tensor = tensor.to(device)
    print(f"🚚 Moving {name}: {old_device} -> {device}")
    return new_tensor


def ensure_same_device(*tensors, target_device: Optional[torch.device] = None, 
                      names: Optional[List[str]] = None) -> List[torch.Tensor]:
    """모든 텐서를 같은 디바이스로 이동"""
    if not tensors:
        return []
    
    if names is None:
        names = [f"tensor_{i}" for i in range(len(tensors))]
    
    # 타겟 디바이스 결정
    if target_device is None:
        target_device = tensors[0].device
    
    result = []
    for tensor, name in zip(tensors, names):
        if tensor.device != target_device:
            print(f"🚚 Moving {name}: {tensor.device} -> {target_device}")
            result.append(tensor.to(target_device))
        else:
            result.append(tensor)
    
    return result
