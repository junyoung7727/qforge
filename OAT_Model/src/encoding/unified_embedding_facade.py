"""
Unified Embedding Facade
통합 임베딩 시스템을 위한 파사드 패턴 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
# 동적 임포트를 위한 경로 설정
import sys
from pathlib import Path

# 현재 파일의 부모 디렉토리들을 경로에 추가
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
project_root = src_dir.parent
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root / "quantumcommon"))

# modular_quantum_attention 임포트
try:
    from encoding.modular_quantum_attention import AttentionMode, StandardQuantumAttention
except ImportError:
    try:
        from .modular_quantum_attention import AttentionMode, StandardQuantumAttention
    except ImportError:
        from OAT_Model.src.encoding.modular_quantum_attention import AttentionMode, StandardQuantumAttention

# 게이트 레지스트리 임포트
try:
    from gates import QuantumGateRegistry
except ImportError:
    try:
        from quantumcommon.gates import QuantumGateRegistry
    except ImportError:
        print("Warning: Could not import QuantumGateRegistry")
        QuantumGateRegistry = None

# Device debugging utilities
try:
    from utils.device_debug import (
        device_debug_context, debug_tensor_device, debug_model_device,
        validate_tensor_devices, check_device, ensure_same_device
    )
except ImportError:
    try:
        from ..utils.device_debug import (
            device_debug_context, debug_tensor_device, debug_model_device,
            validate_tensor_devices, check_device, ensure_same_device
        )
    except ImportError:
        # Fallback functions if device_debug is not available
        def device_debug_context(*args, **kwargs):
            from contextlib import nullcontext
            return nullcontext()
        def debug_tensor_device(tensor, name, location=""): return tensor
        def debug_model_device(model, name=""): return None
        def validate_tensor_devices(*args, **kwargs): return True
        def check_device(tensor, name=""): return str(tensor.device)
        def ensure_same_device(*tensors, **kwargs): return list(tensors)

class UnifiedEmbeddingFacade(nn.Module):
    """통합 임베딩 시스템 파사드"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize gate registry for unified vocabulary
        self.gate_registry = QuantumGateRegistry()
        self.gate_vocab = self.gate_registry.get_gate_vocab()
        self.vocab_size = len(self.gate_vocab)
        
        # Handle both dictionary and object configs
        if isinstance(config, dict):
            self.d_model = config['d_model']
            self.max_qubits = config['max_qubits']
            self.max_gates = config['max_gates']
            num_heads = config['num_heads']
            dropout = config['dropout']
        else:
            # Handle object config (like EmbeddingConfig)
            self.d_model = config.d_model
            self.max_qubits = getattr(config, 'n_qubits', 20)
            self.max_gates = getattr(config, 'max_seq_len', 100)
            num_heads = getattr(config, 'num_heads', 8)
            dropout = getattr(config, 'dropout', 0.1)
        
        # 기본 임베딩 레이어들 - use unified gate vocab size
        self.gate_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.qubit_embedding = nn.Embedding(self.max_qubits, self.d_model)
        self.param_projection = nn.Linear(1, self.d_model)
        
        # 위치 인코딩
        self.position_embedding = nn.Embedding(self.max_gates, self.d_model)
        
        # 어텐션 레이어
        self.attention = StandardQuantumAttention(
            d_model=self.d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 출력 정규화
        self.layer_norm = nn.LayerNorm(self.d_model)
        
    def forward(self, circuit_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            circuit_data: 회로 데이터 딕셔너리
                - gate_types: [batch_size, seq_len] 
                - qubit_indices: [batch_size, seq_len, max_qubits_per_gate]
                - parameters: [batch_size, seq_len, max_params_per_gate]
                
        Returns:
            임베딩 결과 딕셔너리
        """
        batch_size = circuit_data['gate_types'].size(0)
        seq_len = circuit_data['gate_types'].size(1)
        
        # 모델의 device 사용 (embedding layer의 device)
        device = next(self.parameters()).device
        
        # 모든 입력 텐서를 모델과 같은 device로 이동
        for key, value in circuit_data.items():
            if isinstance(value, torch.Tensor):
                circuit_data[key] = value.to(device)
        
        # 게이트 타입 임베딩
        gate_embeds = self.gate_embedding(circuit_data['gate_types'])  # [batch, seq, d_model]
        
        # 큐빗 인덱스 임베딩 (첫 번째 큐빗만 사용)
        if 'qubit_indices' in circuit_data:
            first_qubit = circuit_data['qubit_indices'][:, :, 0]  # [batch, seq]
            first_qubit = torch.clamp(first_qubit, 0, self.max_qubits - 1)
            qubit_embeds = self.qubit_embedding(first_qubit)
        else:
            qubit_embeds = torch.zeros_like(gate_embeds)
        
        # 파라미터 임베딩
        if 'parameters' in circuit_data and circuit_data['parameters'].size(-1) > 0:
            first_param = circuit_data['parameters'][:, :, 0:1]  # [batch, seq, 1]
            param_embeds = self.param_projection(first_param)
        else:
            param_embeds = torch.zeros_like(gate_embeds)
        
        # 위치 인코딩
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        
        # 모든 임베딩 결합
        combined_embeds = gate_embeds + qubit_embeds + param_embeds + pos_embeds
        combined_embeds = self.layer_norm(combined_embeds)
        
        # 어텐션 적용
        attention_output = self.attention(combined_embeds)
        
        # Decision Transformer에 필요한 마스크 생성
        # Causal attention mask: [batch, seq_len, seq_len] 하삼각 행렬
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        attention_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Action prediction mask: [batch, seq_len] 
        action_prediction_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        return {
            'embeddings': attention_output,
            'attention_mask': attention_mask,
            'action_prediction_mask': action_prediction_mask,
            'gate_embeddings': gate_embeds,
            'qubit_embeddings': qubit_embeds,
            'parameter_embeddings': param_embeds,
            'position_embeddings': pos_embeds
        }
    
    def get_circuit_representation(self, embeddings: torch.Tensor) -> torch.Tensor:
        """회로 레벨 표현 생성"""
        # 시퀀스 차원에서 평균 풀링
        return embeddings.mean(dim=1)  # [batch_size, d_model]
