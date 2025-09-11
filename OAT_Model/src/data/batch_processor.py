"""
Batch Processing Module
Handles batch circuit processing with padding and optimization
"""

import torch
import time
from typing import Dict, List, Any
from utils.debug_utils import debug_log, debug_tensor_info


class BatchProcessor:
    """배치 처리 전용 클래스"""
    
    def __init__(self, config, circuit_processor, cache_manager):
        self.config = config
        self.circuit_processor = circuit_processor
        self.cache_manager = cache_manager
    
    def process_batch(self, circuit_specs: List) -> Dict[str, torch.Tensor]:
        """배치 처리 (캐싱 및 메모리 효율성 적용)"""
        debug_log("=== BATCH PROCESSING START ===")
        debug_log(f"Input batch size: {len(circuit_specs)}")
        
        if not circuit_specs:
            debug_log("Empty circuit_specs - returning empty result", "WARN")
            return {}
        
        batch_size = len(circuit_specs)
        
        # 배치 메타데이터 캐시 확인
        circuit_ids = [spec.circuit_id for spec in circuit_specs]
        cached_metadata = self.cache_manager.get_cached_batch_metadata(circuit_ids)
        
        if cached_metadata is not None:
            debug_log("Using cached batch metadata")
            return self._process_batch_with_cached_metadata(circuit_specs, cached_metadata)
        
        # 새로운 배치 처리
        return self._process_batch_from_scratch(circuit_specs)
    
    def _process_batch_from_scratch(self, circuit_specs: List) -> Dict[str, torch.Tensor]:
        """캐시 없이 배치 처리"""
        start_time = time.time()
        batch_size = len(circuit_specs)
        
        # 최대 시퀀스 길이 계산
        max_seq_len = self._calculate_max_sequence_length(circuit_specs)
        debug_log(f"Calculated max_seq_len: {max_seq_len}")
        
        # 각 회로를 개별 처리 (패딩 적용)
        individual_results = []
        for i, circuit_spec in enumerate(circuit_specs):
            debug_log(f"Processing circuit {i+1}/{batch_size}: {circuit_spec.circuit_id}")
            result = self.circuit_processor.process_circuit_with_padding(circuit_spec, max_seq_len)
            individual_results.append(result)
        
        # 배치로 결합
        combined = self._combine_individual_results(individual_results, circuit_specs)
        
        # 액션 마스크 조정
        combined = self._adjust_action_masks(combined, circuit_specs)
        
        # 배치 메타데이터 캐시에 저장
        self.cache_manager.cache_batch_metadata(combined)
        
        debug_log(f"Batch processing completed in {time.time() - start_time:.3f}s")
        debug_log("=== BATCH PROCESSING END ===")
        
        return combined
    
    def _process_batch_with_cached_metadata(self, circuit_specs: List, 
                                          cached_metadata) -> Dict[str, torch.Tensor]:
        """캐시된 메타데이터를 사용한 배치 처리"""
        debug_log("Processing batch with cached metadata")
        
        # 캐시된 메타데이터에서 정보 추출
        max_seq_len = max(cached_metadata.num_gates) * 3  # SAR 길이 기준
        
        # 개별 회로 처리
        individual_results = []
        for circuit_spec in circuit_specs:
            result = self.circuit_processor.process_circuit_with_padding(circuit_spec, max_seq_len)
            individual_results.append(result)
        
        # 배치로 결합
        combined = self._combine_individual_results(individual_results, circuit_specs)
        combined = self._adjust_action_masks(combined, circuit_specs)
        
        return combined
    
    def _calculate_max_sequence_length(self, circuit_specs: List) -> int:
        """배치의 최대 시퀀스 길이 계산"""
        max_gates = max(len(spec.gates) for spec in circuit_specs)
        max_sar_len = max_gates * 3  # SAR 패턴 (State, Action, Reward)
        
        # 설정된 최대 길이로 제한
        return min(max_sar_len, self.config.max_seq_len)
    
    def _combine_individual_results(self, individual_results: List[Dict], 
                                  circuit_specs: List) -> Dict[str, torch.Tensor]:
        """개별 결과를 배치로 결합"""
        debug_log("Combining individual results into batch")
        
        batch_size = len(individual_results)
        combined = {}
        
        # 텐서 결합
        for key in ['input_sequence', 'attention_mask', 'action_prediction_mask']:
            if key in individual_results[0]:
                tensors = [result[key] for result in individual_results]
                combined[key] = torch.cat(tensors, dim=0)
                debug_log(f"Combined {key}: {combined[key].shape}")
        
        # 메타데이터 결합
        combined.update({
            'circuit_id': [spec.circuit_id for spec in circuit_specs],
            'num_qubits': [spec.num_qubits for spec in circuit_specs],
            'num_gates': [len(spec.gates) for spec in circuit_specs],
            'batch_size': batch_size
        })
        
        return combined
    
    def _adjust_action_masks(self, combined: Dict[str, torch.Tensor], 
                           circuit_specs: List) -> Dict[str, torch.Tensor]:
        """각 회로의 실제 게이트 수에 맞게 액션 마스크 조정"""
        if 'action_prediction_mask' not in combined:
            debug_log("No action_prediction_mask to adjust", "WARN")
            return combined
        
        action_mask = combined['action_prediction_mask']
        batch_size = len(circuit_specs)
        gate_counts = [len(spec.gates) for spec in circuit_specs]
        max_sar_len = action_mask.shape[1]
        
        debug_log(f"Adjusting action masks for batch_size={batch_size}, max_sar_len={max_sar_len}")
        debug_log(f"Gate counts: {gate_counts}")
        
        # 각 회로별로 실제 게이트 수만큼만 액션 위치를 True로 설정
        for b in range(batch_size):
            actual_gates = gate_counts[b]
            
            # 전체 마스크를 False로 초기화
            action_mask[b] = False
            
            # 실제 액션 위치만 True로 설정 (1::3 패턴)
            for i in range(actual_gates):
                action_pos = i * 3 + 1  # 1, 4, 7, 10...
                if action_pos < max_sar_len:
                    action_mask[b, action_pos] = True
        
        combined['action_prediction_mask'] = action_mask
        
        # 검증
        total_true = action_mask.sum().item()
        expected_true = sum(gate_counts)
        debug_log(f"Action mask adjustment validation: True count {total_true}, expected {expected_true}")
        
        if total_true != expected_true:
            raise ValueError(f"Action mask adjustment failed! True count: {total_true}, expected: {expected_true}")
        
        return combined
    
    def process_batch_optimized(self, circuit_specs: List, 
                              use_parallel: bool = False) -> Dict[str, torch.Tensor]:
        """최적화된 배치 처리 (선택적 병렬 처리)"""
        if use_parallel and len(circuit_specs) > 4:
            return self._process_batch_parallel(circuit_specs)
        else:
            return self.process_batch(circuit_specs)
    
    def _process_batch_parallel(self, circuit_specs: List) -> Dict[str, torch.Tensor]:
        """병렬 배치 처리 (멀티스레딩)"""
        import concurrent.futures
        import threading
        
        debug_log(f"Processing batch with parallel execution: {len(circuit_specs)} circuits")
        
        max_seq_len = self._calculate_max_sequence_length(circuit_specs)
        
        # 스레드 안전 결과 저장
        results = [None] * len(circuit_specs)
        lock = threading.Lock()
        
        def process_single(idx, spec):
            try:
                result = self.circuit_processor.process_circuit_with_padding(spec, max_seq_len)
                with lock:
                    results[idx] = result
            except Exception as e:
                debug_log(f"Error processing circuit {idx}: {e}", "ERROR")
                with lock:
                    results[idx] = None
        
        # 병렬 실행
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_single, i, spec) 
                      for i, spec in enumerate(circuit_specs)]
            concurrent.futures.wait(futures)
        
        # 실패한 회로 제거
        valid_results = []
        valid_specs = []
        for i, (result, spec) in enumerate(zip(results, circuit_specs)):
            if result is not None:
                valid_results.append(result)
                valid_specs.append(spec)
            else:
                debug_log(f"Skipping failed circuit {i}: {spec.circuit_id}", "WARN")
        
        if not valid_results:
            debug_log("All circuits failed in parallel processing", "ERROR")
            return {}
        
        # 배치로 결합
        combined = self._combine_individual_results(valid_results, valid_specs)
        combined = self._adjust_action_masks(combined, valid_specs)
        
        return combined
