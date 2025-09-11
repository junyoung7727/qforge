"""
Refactored Embedding Pipeline Module
Clean, modular implementation using separated components
"""

import torch
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Import utilities
from utils.debug_utils import debug_print, debug_log, debug_tensor_info

# Import modular components
from .cache_manager import CacheManager
from .circuit_processor import CircuitProcessor
from .batch_processor import BatchProcessor

# Import dependencies with fallback paths
try:
    from .quantum_circuit_dataset import CircuitSpec
    from ..encoding.grid_graph_encoder import GridGraphEncoder
except ImportError:
    try:
        from .quantum_circuit_dataset import CircuitSpec
        from ..encoding.grid_graph_encoder import GridGraphEncoder
    except ImportError:
        sys.path.append(str(Path(__file__).parent.parent))
        from data.quantum_circuit_dataset import CircuitSpec
        from encoding.grid_graph_encoder import GridGraphEncoder

# Gate registry import
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from gates import QuantumGateRegistry


@dataclass
class EmbeddingConfig:
    """임베딩 설정"""
    d_model: int = 512
    n_gate_types: int = None  # 게이트 vocab 싱글톤에서 자동 설정
    n_qubits: int = 50
    max_seq_len: int = 1000
    max_time_steps: int = 150
    device: str = 'cuda'  # 디바이스 설정 추가

    def __post_init__(self):
        """초기화 후 gate 수를 싱글톤에서 가져오기"""
        if self.n_gate_types is None:
            self.n_gate_types = QuantumGateRegistry.get_singleton_gate_count()
            print(f"EmbeddingConfig: Using gate vocab singleton, n_gate_types = {self.n_gate_types}")


class EmbeddingPipeline:
    """리팩토링된 임베딩 파이프라인 - 모듈화된 컴포넌트 사용"""
    
    def __init__(self, config: EmbeddingConfig, enable_cache: bool = True):
        self.config = config
        self.enable_cache = enable_cache
        
        # 게이트 레지스트리 초기화
        self.gate_registry = QuantumGateRegistry()
        self.gate_vocab = self.gate_registry.get_gate_vocab()
        
        # 게이트 수 확인 및 설정 동기화
        actual_gate_count = len(self.gate_vocab)
        if self.config.n_gate_types != actual_gate_count:
            print(f"Config mismatch: expected {self.config.n_gate_types}, got {actual_gate_count}")
            self.config.n_gate_types = actual_gate_count
        print(f"EmbeddingPipeline initialized with {actual_gate_count} gate types (Cache: {enable_cache})")
        
        # 핵심 컴포넌트 초기화
        self._initialize_components()
    
    def _initialize_components(self):
        """모듈화된 컴포넌트들 초기화"""
        # Grid Encoder 초기화
        self.grid_encoder = GridGraphEncoder()
        
        # Unified embedding facade 초기화
        from encoding.unified_embedding_facade import UnifiedEmbeddingFacade
        self.unified_facade = UnifiedEmbeddingFacade(self.config)
        
        # 모듈화된 컴포넌트들 초기화
        self.cache_manager = CacheManager(
            enable_cache=self.enable_cache,
            max_cache_size=1000,
            max_batch_cache_size=800
        )
        
        self.circuit_processor = CircuitProcessor(
            config=self.config,
            grid_encoder=self.grid_encoder,
            unified_facade=self.unified_facade
        )
        
        self.batch_processor = BatchProcessor(
            config=self.config,
            circuit_processor=self.circuit_processor,
            cache_manager=self.cache_manager
        )
    
    def process_single_circuit(self, circuit_spec: CircuitSpec) -> Dict[str, torch.Tensor]:
        """단일 회로 처리 - 깔끔한 인터페이스"""
        return self.circuit_processor.process_circuit(circuit_spec, self.cache_manager)
    
    def process_circuit(self, circuit_spec) -> Dict[str, Any]:
        """단일 회로 처리 - 통합 인터페이스 (하위 호환성)"""
        result = self.process_single_circuit(circuit_spec)
        return {'decision_transformer': result}
    
    def process_batch(self, circuit_specs: List[CircuitSpec]) -> Dict[str, torch.Tensor]:
        """배치 처리 - 모듈화된 배치 프로세서 사용"""
        return self.batch_processor.process_batch(circuit_specs)
    
    def process_batch_optimized(self, circuit_specs: List[CircuitSpec], 
                              use_parallel: bool = False) -> Dict[str, torch.Tensor]:
        """최적화된 배치 처리"""
        return self.batch_processor.process_batch_optimized(circuit_specs, use_parallel)
    
    # 캐시 관리 메서드들 (위임)
    def clear_cache(self) -> None:
        """모든 캐시 초기화"""
        self.cache_manager.clear_cache()
        print("모든 캐시가 초기화되었습니다.")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        return self.cache_manager.get_cache_stats()
    
    def print_cache_stats(self) -> None:
        """캐시 통계 출력"""
        self.cache_manager.print_cache_stats()
    
    def get_cached_batch_metadata(self, circuit_ids: List[str]):
        """배치 메타데이터 캐시 조회 (하위 호환성)"""
        return self.cache_manager.get_cached_batch_metadata(circuit_ids)
    
    # 설정 및 상태 관리
    def get_config(self) -> EmbeddingConfig:
        """현재 설정 반환"""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                print(f"Config updated: {key} = {value}")
            else:
                print(f"Warning: Unknown config key: {key}")
    
    def get_gate_vocab_info(self) -> Dict[str, Any]:
        """게이트 vocabulary 정보 반환"""
        return {
            'gate_vocab': self.gate_vocab,
            'n_gate_types': len(self.gate_vocab),
            'gate_registry': str(self.gate_registry)
        }
    
    def validate_circuit_spec(self, circuit_spec: CircuitSpec) -> bool:
        """회로 스펙 유효성 검증"""
        try:
            if not hasattr(circuit_spec, 'circuit_id'):
                debug_log("Missing circuit_id", "ERROR")
                return False
            
            if not hasattr(circuit_spec, 'gates'):
                debug_log("Missing gates", "ERROR")
                return False
            
            if not hasattr(circuit_spec, 'num_qubits'):
                debug_log("Missing num_qubits", "ERROR")
                return False
            
            return True
        except Exception as e:
            debug_log(f"Circuit spec validation error: {e}", "ERROR")
            return False
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        # Unified embedding facade 초기화
        from encoding.unified_embedding_facade import UnifiedEmbeddingFacade
        self.unified_facade = UnifiedEmbeddingFacade(self.config)
        return {
            'config': {
                'd_model': self.config.d_model,
                'n_gate_types': self.config.n_gate_types,
                'n_qubits': self.config.n_qubits,
                'max_seq_len': self.config.max_seq_len,
                'max_time_steps': self.config.max_time_steps
            },
            'cache_enabled': self.enable_cache,
            'cache_stats': self.get_cache_stats(),
            'gate_vocab_size': len(self.gate_vocab),
            'components': {
                'grid_encoder': str(type(self.grid_encoder).__name__),
                'unified_facade': str(type(self.unified_facade).__name__),
                'cache_manager': str(type(self.cache_manager).__name__),
                'circuit_processor': str(type(self.circuit_processor).__name__),
                'batch_processor': str(type(self.batch_processor).__name__)
            }
        }
    
    def __repr__(self) -> str:
        """파이프라인 문자열 표현"""
        return (f"EmbeddingPipeline(d_model={self.config.d_model}, "
                f"n_gate_types={self.config.n_gate_types}, "
                f"cache_enabled={self.enable_cache})")


# 하위 호환성을 위한 별칭
RefactoredEmbeddingPipeline = EmbeddingPipeline
