"""
Decision Transformer 전용 인코더
증분적 게이트 추가 및 액션 예측에 특화된 GNN 기반 인코더
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import copy

from .gnn_quantum_encoder import GNNQuantumEncoder
from .base_quantum_encoder import EncodingResult, QuantumGraphData

class IncrementalGraphBuilder:
    """증분적 그래프 구축을 위한 빌더"""
    
    def __init__(self):
        self.base_builder = None
        
    def build_incremental_graph(self, circuit_spec, step: int) -> QuantumGraphData:
        """특정 스텝까지의 부분 그래프 생성"""
        # 스텝까지의 게이트만 포함하는 부분 회로 생성
        partial_gates = circuit_spec.gates[:step] if step > 0 else []
        
        # 부분 CircuitSpec 생성
        from ..data.quantum_circuit_dataset import CircuitSpec
        partial_spec = CircuitSpec(
            circuit_id=f"{circuit_spec.circuit_id}_step_{step}",
            gates=partial_gates,
            num_qubits=circuit_spec.num_qubits,
            properties=circuit_spec.properties
        )
        
        # 그래프 구조 생성
        from ..encoding.grid_graph_encoder import GridGraphEncoder
        from .base_quantum_encoder import QuantumCircuitGraphBuilder
        
        grid_encoder = GridGraphEncoder()
        encoded_data = grid_encoder.encode(partial_spec)
        grid_matrix_data = grid_encoder.to_grid_matrix(encoded_data)
        
        if self.base_builder is None:
            self.base_builder = QuantumCircuitGraphBuilder()
        
        return self.base_builder.build_graph_from_grid(grid_matrix_data)

class ActionPredictionHead(nn.Module):
    """액션 예측을 위한 전용 헤드"""
    
    def __init__(self, d_model: int, n_gate_types: int, max_qubits: int = 20):
        super().__init__()
        self.d_model = d_model
        self.n_gate_types = n_gate_types
        self.max_qubits = max_qubits
        
        # 게이트 타입 예측
        self.gate_type_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_gate_types)
        )
        
        # 큐비트 위치 예측 (첫 번째 큐비트)
        self.qubit1_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, max_qubits)
        )
        
        # 큐비트 위치 예측 (두 번째 큐비트)
        self.qubit2_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, max_qubits)
        )
        
        # 파라미터 예측 (회전각 등)
        self.parameter_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, 1),
            nn.Tanh()  # [-1, 1] 범위로 정규화
        )
        
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """액션 예측 수행"""
        return {
            'gate_type': self.gate_type_head(embeddings),
            'qubit1': self.qubit1_head(embeddings),
            'qubit2': self.qubit2_head(embeddings),
            'parameter': self.parameter_head(embeddings)
        }

class DecisionTransformerEncoder(GNNQuantumEncoder):
    """Decision Transformer 전용 GNN 인코더"""
    
    def __init__(self, d_model: int, config: Dict[str, Any]):
        super().__init__(d_model, config)
        
        self.incremental_builder = IncrementalGraphBuilder()
        
        # Decision Transformer 특화 구성요소
        self.max_sequence_length = config.get('max_sequence_length', 1000)
        self.n_gate_types = config.get('n_gate_types', 20)
        self.max_qubits = config.get('max_qubits', 20)
        
        # 액션 예측 헤드
        self.action_head = ActionPredictionHead(d_model, self.n_gate_types, self.max_qubits)
        
        # 상태-액션-보상 임베딩
        self.state_embedding = nn.Linear(d_model, d_model)
        self.action_embedding = nn.Embedding(self.n_gate_types, d_model)
        self.reward_embedding = nn.Linear(1, d_model)
        
        # 시퀀스 위치 임베딩
        self.sequence_pos_embedding = nn.Embedding(self.max_sequence_length, d_model)
        
        # Transformer 디코더 (자기회귀적 예측용)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
    def encode_for_action_prediction(self, circuit_spec, current_step: int) -> EncodingResult:
        """액션 예측을 위한 인코딩"""
        # 현재 스텝까지의 증분적 그래프 생성
        graph_data = self.incremental_builder.build_incremental_graph(circuit_spec, current_step)
        
        # GNN으로 그래프 인코딩
        graph_embedding = self._encode_graph(graph_data)
        
        # 시퀀스 위치 정보 추가
        pos_ids = torch.arange(current_step + 1, device=graph_embedding.device)
        pos_embeddings = self.sequence_pos_embedding(pos_ids.clamp(0, self.max_sequence_length - 1))
        
        # 상태 임베딩과 위치 임베딩 결합
        if graph_embedding.size(0) == 1:  # 단일 그래프
            state_emb = self.state_embedding(graph_embedding)
            # 위치 임베딩을 마지막 위치에 추가
            if len(pos_embeddings) > 0:
                state_emb = state_emb + pos_embeddings[-1:].unsqueeze(0)
        else:
            state_emb = self.state_embedding(graph_embedding)
        
        # 액션 예측
        action_predictions = self.action_head(state_emb)
        
        return EncodingResult(
            embeddings=state_emb,
            graph_data=graph_data,
            auxiliary_data={
                'action_predictions': action_predictions,
                'current_step': current_step,
                'encoding_type': 'decision_transformer'
            }
        )
    
    def encode_sequence(self, circuit_spec, target_properties: Optional[Dict[str, float]] = None) -> EncodingResult:
        """전체 시퀀스 인코딩 (학습용)"""
        sequence_length = len(circuit_spec.gates)
        
        # 각 스텝별 상태 임베딩 생성
        state_embeddings = []
        action_embeddings = []
        
        for step in range(sequence_length + 1):  # +1 for final state
            # 현재 스텝까지의 그래프 인코딩
            graph_data = self.incremental_builder.build_incremental_graph(circuit_spec, step)
            graph_emb = self._encode_graph(graph_data)
            state_emb = self.state_embedding(graph_emb)
            state_embeddings.append(state_emb)
            
            # 액션 임베딩 (다음 게이트 정보)
            if step < sequence_length:
                next_gate = circuit_spec.gates[step]
                gate_type_id = self._get_gate_type_id(next_gate.gate_type)
                action_emb = self.action_embedding(torch.tensor(gate_type_id, device=graph_emb.device))
                action_embeddings.append(action_emb.unsqueeze(0))
        
        # 시퀀스 텐서 생성
        state_sequence = torch.cat(state_embeddings, dim=0)  # [seq_len+1, d_model]
        if action_embeddings:
            action_sequence = torch.cat(action_embeddings, dim=0)  # [seq_len, d_model]
        else:
            action_sequence = torch.zeros(0, self.d_model, device=state_sequence.device)
        
        # 위치 임베딩 추가
        seq_len = state_sequence.size(0)
        pos_ids = torch.arange(seq_len, device=state_sequence.device)
        pos_embeddings = self.sequence_pos_embedding(pos_ids.clamp(0, self.max_sequence_length - 1))
        state_sequence = state_sequence + pos_embeddings
        
        # 보상 정보 (타겟 속성값 기반)
        reward_sequence = self._compute_reward_sequence(circuit_spec, target_properties)
        
        return EncodingResult(
            embeddings=state_sequence.unsqueeze(0),  # [1, seq_len, d_model]
            auxiliary_data={
                'state_sequence': state_sequence,
                'action_sequence': action_sequence,
                'reward_sequence': reward_sequence,
                'sequence_length': sequence_length,
                'encoding_type': 'decision_transformer_sequence'
            }
        )
    
    def _get_gate_type_id(self, gate_type: str) -> int:
        """게이트 타입을 ID로 변환"""
        # 게이트 레지스트리에서 ID 조회
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent.parent.parent / "quantumcommon"))
            from gates import QuantumGateRegistry
            
            registry = QuantumGateRegistry()
            return registry.get_gate_type(gate_type)
        except:
            # 기본 매핑
            gate_map = {
                'i': 0, 'x': 1, 'y': 2, 'z': 3, 'h': 4,
                'rx': 5, 'ry': 6, 'rz': 7, 'cx': 8, 'cy': 9, 'cz': 10,
                'ccx': 11, 'swap': 12, 's': 13, 't': 14, 'sdg': 15, 'tdg': 16
            }
            return gate_map.get(gate_type.lower(), 0)
    
    def _compute_reward_sequence(self, circuit_spec, target_properties: Optional[Dict[str, float]]) -> torch.Tensor:
        """보상 시퀀스 계산"""
        sequence_length = len(circuit_spec.gates)
        
        if target_properties is None:
            # 기본 보상: 회로 완성도 기반
            rewards = torch.linspace(0.0, 1.0, sequence_length + 1)
        else:
            # 타겟 속성 기반 보상 계산
            rewards = torch.zeros(sequence_length + 1)
            
            # 각 스텝에서의 예상 속성값과 타겟 간의 거리 기반 보상
            for step in range(sequence_length + 1):
                progress = step / sequence_length if sequence_length > 0 else 1.0
                
                # 간단한 휴리스틱: 진행도에 따른 선형 보상
                if 'entanglement' in target_properties:
                    target_ent = target_properties['entanglement']
                    expected_ent = progress * target_ent
                    rewards[step] = 1.0 - abs(expected_ent - target_ent)
                else:
                    rewards[step] = progress
        
        return rewards.unsqueeze(-1)  # [seq_len+1, 1]
