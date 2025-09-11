"""
양자 상태 인코더 - 상태 벡터를 임베딩으로 변환하는 전담 클래스
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any

class QuantumStateEncoder(nn.Module):
    """양자 상태 벡터를 임베딩으로 변환"""
    
    def __init__(self, d_model: int = 512, n_gate_types: int = 20):
        super().__init__()
        self.d_model = d_model
        self.n_gate_types = n_gate_types
        
        # 8차원 확장 상태 벡터를 임베딩으로 변환
        # [gate_type, qubit1, qubit2, parameter, time_step, qubit_idx, is_control, is_target]
        self.state_projection = nn.Linear(8, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def encode_states(self, state_vectors: torch.Tensor) -> torch.Tensor:
        """
        상태 벡터들을 임베딩으로 변환
        
        Args:
            state_vectors: [grid_size, 8] 상태 벡터들
            
        Returns:
            [grid_size, d_model] 임베딩
        """
        if state_vectors.size(0) == 0:
            return torch.zeros(1, self.d_model, dtype=torch.float32)
        
        # 선형 변환 + 정규화 + 드롭아웃
        embeddings = self.state_projection(state_vectors)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def encode_single_state(self, gate_type: int, qubit1: int, qubit2: int, 
                           parameter: float, time_step: int, qubit_idx: int,
                           is_control: bool, is_target: bool) -> torch.Tensor:
        """단일 상태를 임베딩으로 변환"""
        state_vector = torch.tensor([
            float(gate_type), float(qubit1), float(qubit2), float(parameter),
            float(time_step), float(qubit_idx), float(is_control), float(is_target)
        ], dtype=torch.float32).unsqueeze(0)  # [1, 8]
        
        return self.encode_states(state_vector).squeeze(0)  # [d_model]
