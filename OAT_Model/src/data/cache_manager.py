"""
Cache Management Module
Handles all caching operations for the embedding pipeline
"""

import torch
import hashlib
import json
import threading
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class BatchMetadata:
    """배치 메타데이터 캐시용 데이터 클래스"""
    circuit_ids: List[str]
    num_qubits: List[int]
    num_gates: List[int]
    batch_size: int
    timestamp: float


class CacheManager:
    """통합 캐시 관리자 - LRU 캐시와 배치 메타데이터 캐시"""
    
    def __init__(self, enable_cache: bool = True, max_cache_size: int = 1000, 
                 max_batch_cache_size: int = 800):
        self.enable_cache = enable_cache
        self._max_cache_size = max_cache_size
        self._max_batch_cache_size = max_batch_cache_size
        
        # 메인 캐시
        self._memory_cache = {}
        self._cache_access_order = []
        self._cache_stats = {'hits': 0, 'misses': 0, 'total': 0}
        
        # 배치 메타데이터 캐시
        self._batch_metadata_cache = {}
        self._batch_cache_access_order = []
        self._batch_cache_stats = {'hits': 0, 'misses': 0, 'total': 0}
        
        # 스레드 안전성
        self._cache_lock = threading.RLock()
    
    def generate_cache_key(self, circuit_spec) -> str:
        """회로 스펙으로부터 고유한 캐시 키 생성"""
        # Handle both dictionary and object circuit specs
        if isinstance(circuit_spec, dict):
            gates = circuit_spec.get('gates', [])
            circuit_id = circuit_spec.get('circuit_id', 'unknown')
            num_qubits = circuit_spec.get('num_qubits', 0)
            depth = circuit_spec.get('depth', 0)
        else:
            gates = getattr(circuit_spec, 'gates', [])
            circuit_id = getattr(circuit_spec, 'circuit_id', 'unknown')
            num_qubits = getattr(circuit_spec, 'num_qubits', 0)
            depth = getattr(circuit_spec, 'depth', 0)
        
        # GateOperation 객체들을 직렬화 가능한 형태로 변환
        serializable_gates = []
        for gate in gates:
            if isinstance(gate, dict):
                # Already a dictionary
                serializable_gates.append({
                    'gate_type': gate.get('gate_type', gate.get('type', 'unknown')),
                    'qubits': gate.get('qubits', []),
                    'parameters': gate.get('parameters', [])
                })
            elif hasattr(gate, '__dict__'):
                # GateOperation 객체인 경우 딕셔너리로 변환
                gate_dict = {
                    'gate_type': getattr(gate, 'gate_type', str(gate)),
                    'qubits': getattr(gate, 'qubits', []),
                    'parameters': getattr(gate, 'parameters', [])
                }
                serializable_gates.append(gate_dict)
            else:
                # 이미 직렬화 가능한 형태인 경우
                serializable_gates.append(str(gate))
        
        circuit_data = {
            'circuit_id': circuit_id,
            'gates': serializable_gates,
            'num_qubits': num_qubits,
            'depth': depth
        }
        data_str = json.dumps(circuit_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get_from_cache(self, cache_key: str) -> Optional[Dict[str, torch.Tensor]]:
        """캐시에서 데이터 조회"""
        if not self.enable_cache:
            return None
            
        with self._cache_lock:
            self._cache_stats['total'] += 1
            
            if cache_key in self._memory_cache:
                cached_data = self._memory_cache[cache_key]
                
                # 캐시 유효성 검증
                if self._is_cache_valid(cached_data):
                    self._update_cache_access(cache_key)
                    self._cache_stats['hits'] += 1
                    return cached_data
                else:
                    # 무효한 캐시 제거
                    self._remove_from_cache(cache_key)
            
            self._cache_stats['misses'] += 1
            return None
    
    def put_to_cache(self, cache_key: str, data: Dict[str, torch.Tensor]) -> None:
        """데이터를 캐시에 저장"""
        if not self.enable_cache:
            return
            
        with self._cache_lock:
            # 캐시 크기 제한 확인
            if len(self._memory_cache) >= self._max_cache_size:
                oldest_key = self._cache_access_order.pop(0)
                del self._memory_cache[oldest_key]
            
            self._memory_cache[cache_key] = data
            self._update_cache_access(cache_key)
    
    def _is_cache_valid(self, cached_data: Dict) -> bool:
        """캐시 데이터 유효성 검증"""
        required_keys = ['num_gates', 'sar_sequence_len', 'original_gate_count']
        return all(key in cached_data for key in required_keys)
    
    def _remove_from_cache(self, cache_key: str) -> None:
        """캐시에서 항목 제거"""
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
        if cache_key in self._cache_access_order:
            self._cache_access_order.remove(cache_key)
    
    def _update_cache_access(self, cache_key: str) -> None:
        """캐시 접근 순서 업데이트 (LRU)"""
        if cache_key in self._cache_access_order:
            self._cache_access_order.remove(cache_key)
        self._cache_access_order.append(cache_key)
    
    def generate_batch_cache_key(self, circuit_ids: List[str]) -> str:
        """배치 메타데이터 캐시 키 생성"""
        sorted_ids = sorted(circuit_ids)
        key_string = "|".join(sorted_ids)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cached_batch_metadata(self, circuit_ids: List[str]) -> Optional[BatchMetadata]:
        """캐시된 배치 메타데이터 조회"""
        if not self.enable_cache:
            return None
            
        cache_key = self.generate_batch_cache_key(circuit_ids)
        
        with self._cache_lock:
            self._batch_cache_stats['total'] += 1
            
            if cache_key in self._batch_metadata_cache:
                self._batch_cache_stats['hits'] += 1
                self._batch_cache_access_order.remove(cache_key)
                self._batch_cache_access_order.append(cache_key)
                return self._batch_metadata_cache[cache_key]
            else:
                self._batch_cache_stats['misses'] += 1
                return None
    
    def cache_batch_metadata(self, batch_data: Dict[str, Any]) -> None:
        """배치 메타데이터 캐시에 저장"""
        if not self.enable_cache:
            return
            
        circuit_ids = batch_data['circuit_id']
        cache_key = self.generate_batch_cache_key(circuit_ids)
        
        with self._cache_lock:
            metadata = BatchMetadata(
                circuit_ids=circuit_ids.copy(),
                num_qubits=batch_data['num_qubits'].copy(),
                num_gates=batch_data['num_gates'].copy(),
                batch_size=len(circuit_ids),
                timestamp=time.time()
            )
            
            self._batch_metadata_cache[cache_key] = metadata
            
            if cache_key in self._batch_cache_access_order:
                self._batch_cache_access_order.remove(cache_key)
            self._batch_cache_access_order.append(cache_key)
            
            # 캐시 크기 제한
            while len(self._batch_metadata_cache) > self._max_batch_cache_size:
                oldest_key = self._batch_cache_access_order.pop(0)
                del self._batch_metadata_cache[oldest_key]
    
    def clear_cache(self) -> None:
        """모든 캐시 초기화"""
        with self._cache_lock:
            self._memory_cache.clear()
            self._cache_access_order.clear()
            self._cache_stats = {'hits': 0, 'misses': 0, 'total': 0}
            
            self._batch_metadata_cache.clear()
            self._batch_cache_access_order.clear()
            self._batch_cache_stats = {'hits': 0, 'misses': 0, 'total': 0}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        with self._cache_lock:
            hit_rate = self._cache_stats['hits'] / max(self._cache_stats['total'], 1) * 100
            batch_hit_rate = self._batch_cache_stats['hits'] / max(self._batch_cache_stats['total'], 1) * 100
            
            return {
                'cache_enabled': self.enable_cache,
                'embedding_cache': {
                    'total_requests': self._cache_stats['total'],
                    'cache_hits': self._cache_stats['hits'],
                    'cache_misses': self._cache_stats['misses'],
                    'hit_rate_percent': hit_rate,
                    'cache_size': len(self._memory_cache),
                },
                'batch_metadata_cache': {
                    'total_requests': self._batch_cache_stats['total'],
                    'cache_hits': self._batch_cache_stats['hits'],
                    'cache_misses': self._batch_cache_stats['misses'],
                    'hit_rate_percent': batch_hit_rate,
                    'cache_size': len(self._batch_metadata_cache),
                },
                'max_cache_size': self._max_cache_size
            }
    
    def print_cache_stats(self) -> None:
        """캐시 통계 출력"""
        stats = self.get_cache_stats()
        print(f"\n캐시 통계:")
        print(f"   - 캐시 활성화: {stats['cache_enabled']}")
        
        embedding_stats = stats['embedding_cache']
        print(f"\n   임베딩 캐시:")
        print(f"      - 총 요청: {embedding_stats['total_requests']}")
        print(f"      - 캐시 히트: {embedding_stats['cache_hits']}")
        print(f"      - 캐시 미스: {embedding_stats['cache_misses']}")
        print(f"      - 히트율: {embedding_stats['hit_rate_percent']:.1f}%")
        print(f"      - 캐시 크기: {embedding_stats['cache_size']}/{stats['max_cache_size']}")
        
        batch_stats = stats['batch_metadata_cache']
        print(f"\n   배치 메타데이터 캐시:")
        print(f"      - 총 요청: {batch_stats['total_requests']}")
        print(f"      - 캐시 히트: {batch_stats['cache_hits']}")
        print(f"      - 캐시 미스: {batch_stats['cache_misses']}")
        print(f"      - 히트율: {batch_stats['hit_rate_percent']:.1f}%")
        print(f"      - 캐시 크기: {batch_stats['cache_size']}")
