"""
Circuit Processing Module
Handles single circuit processing and encoding operations
"""

import torch
import time
from typing import Dict, Any, Optional
from utils.debug_utils import debug_log, debug_tensor_info


class CircuitProcessor:
    """단일 회로 처리 전용 클래스"""
    
    def __init__(self, config, grid_encoder, unified_facade):
        self.config = config
        self.grid_encoder = grid_encoder
        self.unified_facade = unified_facade
    
    def process_circuit(self, circuit_spec, cache_manager) -> Dict[str, torch.Tensor]:
        """단일 회로 처리 (캐싱 적용)"""
        debug_log("=== SINGLE CIRCUIT PROCESSING START ===")
        
        # 캐시 키 생성 및 조회
        cache_key = cache_manager.generate_cache_key(circuit_spec)
        debug_log(f"Generated cache key: {cache_key[:50]}...")
        
        cached_result = cache_manager.get_from_cache(cache_key)
        if cached_result is not None:
            debug_log("Cache HIT - returning cached result")
            debug_tensor_info("cached_result", cached_result, detailed=True)
            return cached_result
        
        # 캐시 미스 - 새로 계산
        start_time = time.time()
        result = self._process_circuit_from_scratch(circuit_spec)
        
        # 캐시에 저장
        cache_manager.put_to_cache(cache_key, result)
        debug_log(f"Result cached. Compute time: {time.time() - start_time:.3f}s")
        
        debug_log("=== SINGLE CIRCUIT PROCESSING END ===")
        return result
    
    def _process_circuit_from_scratch(self, circuit_spec) -> Dict[str, torch.Tensor]:
        """캐시 없이 회로 처리"""
        # 1. Grid Encoder로 회로 인코딩
        encoded_data = self.grid_encoder.encode(circuit_spec)
        debug_tensor_info("encoded_data", encoded_data, detailed=True)
        
        # 2. 인코딩된 데이터를 그리드 매트릭스로 변환
        grid_matrix_data = self.grid_encoder.to_grid_matrix(encoded_data)
        debug_tensor_info("grid_matrix_data", grid_matrix_data, detailed=True)
        
        # 3. 통합 임베딩 Facade 사용
        original_gate_count = len(circuit_spec.gates)
        
        # Convert grid matrix data to facade input format
        facade_input = self._convert_grid_to_facade_input(grid_matrix_data)
        debug_tensor_info("facade_input", facade_input, detailed=True)
        
        dt_results = self.unified_facade(facade_input)
        debug_tensor_info("dt_results", dt_results, detailed=True)
        
        # 4. 배치 차원 추가 및 마스크 처리
        dt_results = self._add_batch_dimensions(dt_results)
        
        # 5. attention_mask 검증 (통합 임베딩에서 제공되어야 함)
        if 'attention_mask' in dt_results:
            debug_log(f"✅ attention_mask provided by facade: {dt_results['attention_mask'].shape}")
        else:
            debug_log("⚠️ attention_mask missing from facade - this should not happen")
            # Fallback: 생성
            if 'input_sequence' in dt_results:
                seq_len = dt_results['input_sequence'].shape[1]
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
                dt_results['attention_mask'] = causal_mask.unsqueeze(0)
                debug_log(f"Generated fallback attention_mask: {dt_results['attention_mask'].shape}")
        
        # 6. targets 생성 (근본적 해결)
        dt_results = self._generate_targets(dt_results, circuit_spec)
        
        # 7. 메타데이터 추가
        dt_results = self._add_metadata(dt_results, circuit_spec, original_gate_count)
        
        return dt_results
    
    def _generate_targets(self, dt_results: Dict[str, torch.Tensor], circuit_spec) -> Dict[str, torch.Tensor]:
        """근본적 해결: 회로 스펙으로부터 실제 targets 생성"""
        debug_log("Generating targets from circuit specification")
        
        # 회로의 게이트 정보로부터 targets 생성
        gates = circuit_spec.gates
        seq_len = dt_results['embeddings'].shape[1] if 'embeddings' in dt_results else len(gates)
        
        # Decision Transformer를 위한 next gate prediction targets
        # 각 위치에서 다음 게이트를 예측하도록 targets 설정
        gate_types = []
        qubit_targets = []
        param_targets = []
        
        for i, gate in enumerate(gates):
            # Gate type encoding (gate registry 사용)
            gate_type_id = self.config.gate_registry.get_gate_type(gate.gate_type) if hasattr(self.config, 'gate_registry') else 0
            gate_types.append(gate_type_id)
            
            # Qubit targets (첫 번째 큐빗만 사용)
            qubit_id = gate.qubits[0] if gate.qubits else 0
            qubit_targets.append(qubit_id)
            
            # Parameter targets (첫 번째 파라미터만 사용)
            param_val = gate.parameters[0] if gate.parameters else 0.0
            param_targets.append(param_val)
        
        # 텐서로 변환
        import torch
        device = dt_results['embeddings'].device if 'embeddings' in dt_results else torch.device('cpu')
        
        # Pad sequences to match embedding length
        while len(gate_types) < seq_len:
            gate_types.append(0)  # Padding token
            qubit_targets.append(0)
            param_targets.append(0.0)
        
        # Truncate if longer
        gate_types = gate_types[:seq_len]
        qubit_targets = qubit_targets[:seq_len]
        param_targets = param_targets[:seq_len]
        
        # Create target tensors
        targets = {
            'gate_types': torch.tensor(gate_types, dtype=torch.long, device=device).unsqueeze(0),
            'qubits': torch.tensor(qubit_targets, dtype=torch.long, device=device).unsqueeze(0),
            'parameters': torch.tensor(param_targets, dtype=torch.float, device=device).unsqueeze(0)
        }
        
        # Stack into single target tensor for compatibility
        # [batch_size, seq_len, target_dim] where target_dim = gate_type + qubit + param
        combined_targets = torch.stack([
            targets['gate_types'].float(),
            targets['qubits'].float(),
            targets['parameters']
        ], dim=-1)  # [1, seq_len, 3]
        
        dt_results['targets'] = combined_targets
        dt_results['target_details'] = targets
        
        debug_log(f"Generated targets shape: {combined_targets.shape}")
        debug_log(f"Target device: {combined_targets.device}")
        
        return dt_results
    
    def _convert_grid_to_facade_input(self, grid_matrix_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert grid matrix data to unified facade input format"""
        # Extract gate types, qubit indices, and parameters from grid matrix data
        # This is a simplified conversion - may need adjustment based on actual grid matrix format
        
        if 'gate_types' in grid_matrix_data:
            gate_types = grid_matrix_data['gate_types']
        else:
            # If no gate_types in grid data, create dummy data
            seq_len = grid_matrix_data.get('sequence_length', 10)
            gate_types = torch.zeros(1, seq_len, dtype=torch.long)
        
        if 'qubit_indices' in grid_matrix_data:
            qubit_indices = grid_matrix_data['qubit_indices']
        else:
            # Create dummy qubit indices
            seq_len = gate_types.size(1) if gate_types.dim() > 1 else 10
            qubit_indices = torch.zeros(1, seq_len, 2, dtype=torch.long)  # max 2 qubits per gate
        
        if 'parameters' in grid_matrix_data:
            parameters = grid_matrix_data['parameters']
        else:
            # Create dummy parameters
            seq_len = gate_types.size(1) if gate_types.dim() > 1 else 10
            parameters = torch.zeros(1, seq_len, 1, dtype=torch.float32)  # max 1 param per gate
        
        return {
            'gate_types': gate_types,
            'qubit_indices': qubit_indices,
            'parameters': parameters
        }
    
    def _add_batch_dimensions(self, dt_results: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decision Transformer를 위한 배치 차원 추가"""
        # Handle facade output format (embeddings)
        if 'embeddings' in dt_results:
            embeddings = dt_results['embeddings']
            # Facade already returns [batch, seq, d_model], so we use it as input_sequence
            dt_results['input_sequence'] = embeddings
            debug_log(f"Using facade embeddings as input_sequence: {embeddings.shape}")
            
            # 🔧 FIX: Facade에서 제공하는 모든 마스크들을 보존
            # attention_mask와 action_prediction_mask가 이미 올바른 차원으로 제공됨
            if 'attention_mask' in dt_results:
                debug_log(f"Preserving facade attention_mask: {dt_results['attention_mask'].shape}")
            if 'action_prediction_mask' in dt_results:
                debug_log(f"Preserving facade action_prediction_mask: {dt_results['action_prediction_mask'].shape}")
        
        # Legacy format handling
        if 'input_sequence' in dt_results and 'embeddings' not in dt_results:
            input_seq = dt_results['input_sequence']
            if len(input_seq.shape) == 2:  # [seq_len, d_model] -> [1, seq_len, d_model]
                dt_results['input_sequence'] = input_seq.unsqueeze(0)
                debug_log(f"Added batch dimension to input_sequence: {input_seq.shape} -> {dt_results['input_sequence'].shape}")
        
        if 'attention_mask' in dt_results:
            attn_mask = dt_results['attention_mask']
            if len(attn_mask.shape) == 2:  # [seq_len, seq_len] -> [1, seq_len, seq_len]
                dt_results['attention_mask'] = attn_mask.unsqueeze(0)
                debug_log(f"Added batch dimension to attention_mask: {attn_mask.shape} -> {dt_results['attention_mask'].shape}")
        
        if 'action_prediction_mask' in dt_results:
            action_mask = dt_results['action_prediction_mask']
            if len(action_mask.shape) == 1:  # [seq_len] -> [1, seq_len]
                dt_results['action_prediction_mask'] = action_mask.unsqueeze(0)
                debug_log(f"Added batch dimension to action_prediction_mask: {action_mask.shape} -> {dt_results['action_prediction_mask'].shape}")
        
        return dt_results
    
    def _add_metadata(self, dt_results: Dict[str, torch.Tensor], circuit_spec, 
                     original_gate_count: int) -> Dict[str, torch.Tensor]:
        """메타데이터 추가"""
        # SAR 시퀀스 길이는 디버깅용으로만 보존
        sar_sequence_len = dt_results.get('sar_sequence_len', original_gate_count * 3)
        if hasattr(sar_sequence_len, 'item'):  # 텐서인 경우 스칼라로 변환
            sar_sequence_len = sar_sequence_len.item()
        
        dt_results.update({
            'circuit_id': circuit_spec.circuit_id,
            'num_qubits': circuit_spec.num_qubits,
            'num_gates': original_gate_count,  # 실제 액션 수 (원래 게이트 수) 사용
            'original_gate_count': original_gate_count,  # 원본 게이트 수
            'sar_sequence_len': sar_sequence_len  # SAR 시퀀스 길이 (디버깅용)
        })
        
        return dt_results
    
    def process_circuit_with_padding(self, circuit_spec, max_seq_len: int) -> Dict[str, torch.Tensor]:
        """패딩이 적용된 단일 회로 처리"""
        debug_log(f"Processing single circuit with padding (max_seq_len={max_seq_len})")
        
        # 기본 처리
        result = self._process_circuit_from_scratch(circuit_spec)
        
        # 패딩 적용
        result = self._apply_padding_to_single_circuit(result, max_seq_len)
        
        return result
    
    def _apply_padding_to_single_circuit(self, result: Dict[str, torch.Tensor], 
                                       max_seq_len: int) -> Dict[str, torch.Tensor]:
        """단일 회로 결과에 패딩 적용"""
        if 'input_sequence' in result:
            input_seq = result['input_sequence']
            if input_seq.shape[1] < max_seq_len:
                # 패딩 적용
                pad_length = max_seq_len - input_seq.shape[1]
                padding = torch.zeros(input_seq.shape[0], pad_length, input_seq.shape[2], 
                                    device=input_seq.device, dtype=input_seq.dtype)
                result['input_sequence'] = torch.cat([input_seq, padding], dim=1)
                debug_log(f"Applied padding to input_sequence: {input_seq.shape} -> {result['input_sequence'].shape}")
        
        if 'attention_mask' in result:
            attn_mask = result['attention_mask']
            if attn_mask.shape[1] < max_seq_len:
                # 어텐션 마스크 패딩
                pad_length = max_seq_len - attn_mask.shape[1]
                
                # 기존 마스크 확장
                padded_mask = torch.zeros(attn_mask.shape[0], max_seq_len, max_seq_len, 
                                        device=attn_mask.device, dtype=attn_mask.dtype)
                padded_mask[:, :attn_mask.shape[1], :attn_mask.shape[2]] = attn_mask
                
                result['attention_mask'] = padded_mask
                debug_log(f"Applied padding to attention_mask: {attn_mask.shape} -> {result['attention_mask'].shape}")
        
        if 'action_prediction_mask' in result:
            action_mask = result['action_prediction_mask']
            if action_mask.shape[1] < max_seq_len:
                # 액션 마스크 패딩 (False로 패딩)
                pad_length = max_seq_len - action_mask.shape[1]
                padding = torch.zeros(action_mask.shape[0], pad_length, 
                                    device=action_mask.device, dtype=action_mask.dtype)
                result['action_prediction_mask'] = torch.cat([action_mask, padding], dim=1)
                debug_log(f"Applied padding to action_prediction_mask: {action_mask.shape} -> {result['action_prediction_mask'].shape}")
        
        return result
