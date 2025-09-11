"""
Decision Transformer Model
간단하고 확장성 높은 Decision Transformer 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np

# Focal Loss for addressing class imbalance in classification tasks
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in classification tasks"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, ignore_index: int = -100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, C] logits
            targets: [N] class indices
        """
        # 기본 CrossEntropy 계산
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        
        # p_t 계산 (정답 클래스에 대한 확률)
        pt = torch.exp(-ce_loss)
        
        # Focal Loss 계산: α * (1-p_t)^γ * CE_loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

from dataclasses import dataclass
from pathlib import Path

# 동적 임포트를 위한 경로 설정
import sys
from pathlib import Path

# 현재 파일의 부모 디렉토리들을 경로에 추가
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
project_root = src_dir.parent.parent
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root / "quantumcommon"))

# 모듈러 어텐션 시스템 임포트
try:
    from encoding.modular_quantum_attention import ModularQuantumAttention, AttentionMode
except ImportError:
    try:
        from ..encoding.modular_quantum_attention import ModularQuantumAttention, AttentionMode
    except ImportError:
        from OAT_Model.src.encoding.modular_quantum_attention import ModularQuantumAttention, AttentionMode

# Property Prediction 모델 임포트
try:
    from models.unified_property_prediction_transformer import UnifiedPropertyPredictionTransformer
except ImportError:
    try:
        from .unified_property_prediction_transformer import UnifiedPropertyPredictionTransformer
    except ImportError:
        from OAT_Model.src.models.unified_property_prediction_transformer import UnifiedPropertyPredictionTransformer

# 게이트 레지스트리 임포트
try:
    from gates import QuantumGateRegistry
except ImportError:
    try:
        from quantumcommon.gates import QuantumGateRegistry
    except ImportError:
        print("Warning: Could not import QuantumGateRegistry")
        QuantumGateRegistry = None

# Debug utilities import
try:
    from utils.debug_utils import debug_print
except ImportError:
    try:
        from ..utils.debug_utils import debug_print
    except ImportError:
        def debug_print(*args, **kwargs):
            print(*args, **kwargs)

# 🗑️ REMOVED: Legacy MultiHeadAttention class - now using ModularAttention system


class TransformerBlock(nn.Module):
    """트랜스포머 블록 (모듈러 양자 어텐션 + 크로스 어텐션 지원)"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, 
                 attention_mode: str = "standard", enable_cross_attention: bool = False):
        super().__init__()
        
        # 모듈러 양자 어텐션 사용 (Self-Attention)
        mode = AttentionMode.ADVANCED if attention_mode == "advanced" else AttentionMode.STANDARD
        self.self_attention = ModularQuantumAttention(
            d_model=d_model,
            num_heads=n_heads,
            mode=mode
        )
        self.attention_mode = attention_mode
        self.enable_cross_attention = enable_cross_attention
        
        # Cross-Attention 추가 (선택적)
        if enable_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm_cross = nn.LayerNorm(d_model, eps=1e-6)
            self.dropout_cross = nn.Dropout(dropout)
        
        # 문제 6 해결: 피드포워드 정규화 최적화 (과도한 정규화 제거)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),  # 중간 dropout만 유지
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Pre-norm 구조 (안정적 epsilon)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 문제 4 해결: 학습 가능한 스케일 파라미터 복원
        self.scale = nn.Parameter(torch.ones(1) * 0.5)  # 학습 가능한 파라미터
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor, 
                graph_data: Optional[Dict] = None, encoder_output: Optional[torch.Tensor] = None,
                encoder_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Remove debug prints that access tensor values during training
        
        # Device synchronization - ensure x is on same device as model
        model_device = next(self.parameters()).device
        if x.device != model_device:
            # Moving x to model device
            x = x.to(model_device)
        
        # Also ensure mask is on same device
        if mask is not None and mask.device != model_device:
            # Moving mask to model device
            mask = mask.to(model_device)
        
        # 1. Self-Attention
        norm_x = self.norm1(x)
        # Remove debug prints that access tensor values during training
        
        # 모듈러 양자 어텐션 적용 (Self-Attention)
        self_attn_out = self.self_attention(norm_x, graph_data=graph_data)
        # Remove debug prints that access tensor values during training
        
        dropout_self_attn = self.dropout1(self_attn_out)
        scaled_self_attn = self.scale * dropout_self_attn
        x = x + scaled_self_attn
        # Remove debug prints that access tensor values during training
        
        # 2. Cross-Attention (선택적)
        if self.enable_cross_attention and encoder_output is not None:
            norm_x_cross = self.norm_cross(x)
            # Cross-attention processing
            
            cross_attn_out, cross_attn_weights = self.cross_attention(
                query=norm_x_cross,
                key=encoder_output,
                value=encoder_output,
                key_padding_mask=encoder_mask,
                need_weights=True
            )
            # Remove debug prints that access tensor values during training
            
            dropout_cross_attn = self.dropout_cross(cross_attn_out)
            scaled_cross_attn = self.scale * dropout_cross_attn
            x = x + scaled_cross_attn
            # Remove debug prints that access tensor values during training
        
        # 3. Feed Forward
        norm_x = self.norm2(x)
        # Remove debug prints that access tensor values during training
        
        ff_out = self.feed_forward(norm_x)
        # Remove debug prints that access tensor values during training
        
        dropout_ff = self.dropout2(ff_out)
        scaled_ff = self.scale * dropout_ff
        x = x + scaled_ff
        # Remove debug prints that access tensor values during training
        
        return x
    
    def set_attention_mode(self, mode: str):
        """어텐션 모드 변경"""
        attention_mode = AttentionMode.ADVANCED if mode.lower() == "advanced" else AttentionMode.STANDARD
        self.attention.set_mode(attention_mode)
        self.attention_mode = mode
        debug_print(f"TransformerBlock attention mode changed to: {mode}")


class DecisionTransformer(nn.Module):
    """Decision Transformer 모델"""
    
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        n_gate_types: int = None,  # 🎆 NEW: gate vocab 싱글톤에서 자동 설정
        max_qubits: int = 50,  # 🎆 NEW: 최대 큐빗 수
        position_dim: int = None,  # 🎆 NEW: 위치 예측 출력 차원 (체크포인트 호환성용)
        dropout: float = 0.1,
        attention_mode: str = "advanced",  # 🎆 NEW: 어텐션 모드 선택
        device: str = "cpu",  # 🎆 NEW: 모델 디바이스 설정
        property_prediction_model: Optional[UnifiedPropertyPredictionTransformer] = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_qubits = max_qubits
        self.device = device
        
        # Use the position_dim from the parameter if provided, otherwise use default
        if position_dim is not None:
            self.position_dim = position_dim
        else:
            self.position_dim = d_model // 4
        
        # 🎆 Gate vocabulary configuration:
        # - Embedding: 22 gates (including special tokens [EOS], [PAD])
        # - Prediction head: 20 gates (excluding special tokens)
        self.gate_registry = QuantumGateRegistry()
        
        # Full vocab for embedding (22 gates including special tokens)
        self.n_embedding_gates = len(self.gate_registry.get_full_gate_vocab())
        
        # Reduced vocab for prediction (20 gates excluding special tokens)  
        self.n_prediction_gates = len(self.gate_registry.get_gate_vocab())
        
        # Use prediction gates for action heads
        if n_gate_types is None:
            self.n_gate_types = self.n_prediction_gates  # 20 gates for prediction
            debug_print(f"🎆 DecisionTransformer: Embedding={self.n_embedding_gates}, Prediction={self.n_prediction_gates}")
        else:
            self.n_gate_types = n_gate_types
            debug_print(f" DecisionTransformer: Using manual n_gate_types = {self.n_gate_types}")
        
        self.attention_mode = attention_mode  # NEW: 어텐션 모드 저장
        
        # 트랜스포머 블록들 생성 (Causal Decoder-Only 구조)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout, attention_mode,
                enable_cross_attention=True  # 인과적 디코더만 사용
            )
            for i in range(n_layers)
        ])
        
        # Cross-attention for integrating quantum circuit constraints
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attention_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.cross_attention_dropout = nn.Dropout(dropout)
        
        debug_print(f"🎆 DecisionTransformer: Gate prediction with property-based RTG rewards")
        
        # Action prediction heads (게이트, 위치, 파라미터) - RESTORED
        self.action_heads = nn.ModuleDict({
            'gate': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, self.n_gate_types)  # 분류: 게이트 타입 예측
            ),
            'position': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(), 
                nn.Linear(d_model // 2, 2 * self.position_dim)  # 분류: [qubit1, qubit2] 예측
            ),
            'parameter': nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1)  # 회귀: 게이트 파라미터 예측
            )
        })
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 모델 초기화 (매우 보수적)
        self.apply(self._conservative_init_weights)
        
        # Cross-attention 시각화 플래그
        self.visualize_cross_attention = False
        
    def _conservative_init_weights(self, module):
        """문제 5 해결: 최적화된 가중치 초기화 (안정성 + 학습 능력)"""
        if isinstance(module, nn.Linear):
            # 문제 5 해결: gain 0.1 → 0.5로 증가 (그래디언트 소실 방지)
            torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 임베딩 초기화도 약간 증가
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        action_prediction_mask: torch.Tensor,
        circuit_constraints: Optional[torch.Tensor] = None,
        num_qubits: Optional[List[int]] = None,
        num_gates: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Clean Decision Transformer Forward Pass
        SAR 시퀀스는 배치 콜레이터에서 이미 구성됨
        
        Args:
            input_sequence: [batch, sar_seq_len, d_model] - 이미 SAR 구성된 시퀀스
            attention_mask: [batch, sar_seq_len, sar_seq_len]
            action_prediction_mask: [batch, sar_seq_len] - Action 위치만 True
        
        Returns:
            Property predictions for action positions
        """
        # Remove debug prints that access tensor values during training
        
        # 디바이스 동기화
        device = input_sequence.device
        attention_mask = attention_mask.to(device)
        action_prediction_mask = action_prediction_mask.to(device)
        
        batch_size, seq_len, _ = input_sequence.shape
        
        # 입력 시퀀스 처리
        x = self.dropout(input_sequence)
        
        # 그래프 데이터 처리 (양자 회로 제약 정보) - 디바이스 동기화
        graph_data = None
        if circuit_constraints is not None:
            circuit_constraints = circuit_constraints.to(device)  # 디바이스 동기화
            graph_data = {
                'num_qubits': num_qubits,
                'num_gates': num_gates
            }
        
        # Transformer blocks with causal attention
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, attention_mask, graph_data)
            
            # Optional circuit constraints integration
            if circuit_constraints is not None:
                x, _ = self.integrate_quantum_constraints(
                    sequence_features=x,
                    constraint_features=circuit_constraints
                )
        
        # Gate prediction heads - predict next gate at action positions
        predictions = {}
        for head_name, head in self.action_heads.items():
            predictions[head_name] = head(x)  # [batch, seq_len, output_dim]
        
        return predictions
    
    
    def _apply_dynamic_qubit_masking(
        self, 
        predictions: Dict[str, torch.Tensor], 
        num_qubits: List[int], 
        action_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        🚀 배치별 큐빗 수에 따른 동적 마스킹 적용
        
        Args:
            predictions: 모델 예측 결과
            num_qubits: 각 회로의 큐빗 수 [batch_size]
            action_mask: 액션 마스크 [batch, num_actions]
        
        Returns:
            마스킹이 적용된 예측 결과
        """
        batch_size = len(num_qubits)
        position_preds = predictions['position_preds']  # [batch, num_actions, max_qubits, 2]
        
        # 배치 크기 검증
        if position_preds.shape[0] != batch_size:
            raise ValueError(
                f"❌ CRITICAL ERROR: 배치 크기 불일치!\n"
                f"   position_preds batch size: {position_preds.shape[0]}\n"
                f"   num_qubits length: {batch_size}"
            )
        
        # 각 회로별로 동적 마스킹 적용
        for batch_idx, circuit_qubits in enumerate(num_qubits):
            # 큐빗 수 검증
            if circuit_qubits <= 0 or circuit_qubits > self.max_qubits:
                raise ValueError(
                    f"❌ CRITICAL ERROR: 회로 {batch_idx}의 큐빗 수가 잘못되었습니다!\n"
                    f"   circuit_qubits: {circuit_qubits}\n"
                    f"   max_qubits: {self.max_qubits}"
                )
            
            # 유효하지 않은 큐빗 인덱스를 -inf로 마스킹 (softmax에서 확률 0)
            # position_preds[batch_idx, :, circuit_qubits:, :] = float('-inf')
            if circuit_qubits < self.max_qubits:
                position_preds[batch_idx, :, circuit_qubits:, :] = float('-inf')
        
        predictions['position_preds'] = position_preds
        return predictions
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict,
        action_prediction_mask: torch.Tensor,
        num_qubits: Optional[List[int]] = None,
        num_gates: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        🎯 별도의 손실 계산 메서드 (예측과 분리)
        
        Args:
            predictions: 모델 예측 결과
            targets: 타겟 데이터
            action_prediction_mask: 액션 예측 마스크
            num_qubits: 배치별 큐빗 수 정보
        
        Returns:
            손실 계산 결과 dict
        """
        loss_computer = ActionLossComputer()
        
        # EOS 마스크 생성 (필요시)
        # Note: For decision transformer, we use reduced vocab (no special tokens)
        # So EOS tokens should not appear in targets, but we keep this for safety
        eos_mask = None
        
        # 마스크 결합
        combined_mask = action_prediction_mask
        if eos_mask is not None:
            combined_mask = action_prediction_mask & eos_mask
        
        # 손실 계산
        return loss_computer.compute(
            predictions=predictions,
            targets=targets,
            mask=combined_mask,
            num_qubits=num_qubits,
            num_gates=num_gates
        )
    
    def set_attention_mode(self, mode: str):
        """모든 트랜스포머 블록의 어텐션 모드 변경"""
        self.attention_mode = mode
        for block in self.transformer_blocks:
            block.set_attention_mode(mode)
        debug_print(f"DecisionTransformer attention mode changed to: {mode}")
    
    def enable_cross_attention_visualization(self, enable: bool = True):
        """크로스 어텐션 시각화 활성화/비활성화"""
        self.visualize_cross_attention = enable
        if enable:
            debug_print("Cross-attention visualization enabled")
        else:
            debug_print("Cross-attention visualization disabled")
    
    def integrate_quantum_constraints(self, 
                                   sequence_features: torch.Tensor,
                                   constraint_features: torch.Tensor,
                                   constraint_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        양자 회로 제약 정보를 시퀀스 특징과 통합
        
        Args:
            sequence_features: [batch, seq_len, d_model] - 시퀀스 특징
            constraint_features: [batch, constraint_len, d_model] - 제약 특징
            constraint_mask: [batch, constraint_len] - 제약 마스크
        
        Returns:
            통합된 시퀀스 특징
        """
        # Pre-norm for cross-attention
        sequence_norm = self.cross_attention_norm(sequence_features)
        
        # Cross-attention: query=sequence, key=value=constraints
        integrated_features, attention_weights = self.cross_attention(
            query=sequence_norm,
            key=constraint_features,
            value=constraint_features,
            key_padding_mask=constraint_mask,
            need_weights=True
        )
        
        # Residual connection with dropout
        output = sequence_features + self.cross_attention_dropout(integrated_features)
        
        debug_print(f"Quantum constraints integrated: seq_shape={sequence_features.shape}, constraint_shape={constraint_features.shape}")
        
        return output, attention_weights
    
    def get_attention_mode(self) -> str:
        """현재 어텐션 모드 반환"""
        return self.attention_mode
    
    def compare_attention_modes(self, input_sequence: torch.Tensor, attention_mask: torch.Tensor, 
                              action_prediction_mask: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """어텐션 모드별 결과 비교"""
        original_mode = self.attention_mode
        results = {}
        
        for mode in ["standard", "advanced", "hybrid"]:
            self.set_attention_mode(mode)
            with torch.no_grad():
                output = self.forward(input_sequence, attention_mask, action_prediction_mask)
                results[mode] = {
                    'action_logits': output['action_logits'].clone(),
                    'action_predictions': output['action_predictions'].clone()
                }
        
        # 원래 모드로 복구
        self.set_attention_mode(original_mode)
    def predict_next_action(
        self,
        input_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        circuit_constraints: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """다음 액션 예측 (추론용) - 멀티태스크 예측 지원
        기대 형태:
          - input_sequence: [1, seq_len, d_model]
          - attention_mask: [1, seq_len, seq_len]
        """
        with torch.no_grad():
            # 마지막 위치에서 액션 예측
            seq_len = input_sequence.shape[1]
            action_prediction_mask = torch.zeros(input_sequence.shape[:2], dtype=torch.bool, device=input_sequence.device)
            for j in range(1, seq_len, 3):
                if j < seq_len:
                    action_prediction_mask[0, j] = True
            action_prediction_mask[0, -1] = True  # 마지막 위치만 예측
            
            outputs = self.forward(input_sequence, attention_mask, action_prediction_mask, circuit_constraints=circuit_constraints)
            
            # 마지막 액션 위치의 확률 분포 반환
            action_positions = torch.where(action_prediction_mask)
            last_action_pos = int(action_positions[-1])
            
            # Property predictions from the cleaned model
            property_preds = outputs[:, last_action_pos, :]  # [1, 3]
            
            return {
                'property_predictions': property_preds,
                'hidden_states': outputs
            }
    
    def sample_gate_with_constraints(
        self, 
        gate_probs: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[int, str, int, int]:
        """게이트 타입을 샘플링하고 해당 게이트의 큐빗/파라미터 요구사항 반환
        
        Returns:
            gate_idx: 샘플링된 게이트 인덱스
            gate_name: 게이트 이름
            required_qubits: 필요한 큐빗 수
            required_params: 필요한 파라미터 수
        """
        # 온도 스케일링
        if temperature != 1.0:
            gate_probs = gate_probs / temperature
            gate_probs = F.softmax(gate_probs, dim=-1)
        
        # 게이트 샘플링
        gate_idx = torch.multinomial(gate_probs.squeeze(0), 1)
        
        # 게이트 정보 조회
        gate_vocab = self.gate_registry.get_gate_vocab()
        gate_names = list(gate_vocab.keys())
        
        if gate_idx < len(gate_names):
            gate_name = gate_names[gate_idx]
            gate_def = self.gate_registry.get_gate(gate_name)
            
            if gate_def:
                return gate_idx, gate_name, gate_def.num_qubits, gate_def.num_parameters
        
        # 기본값 (알 수 없는 게이트)
        return gate_idx, "unknown", 1, 0
    
    def sample_qubits_for_gate(
        self,
        qubit_probs: List[torch.Tensor],
        gate_name: str,
        num_qubits_required: int,
        available_qubits: int,
        temperature: float = 1.0
    ) -> List[int]:
        """게이트 요구사항에 맞는 큐빗 위치들을 샘플링
        
        Args:
            qubit_probs: 각 큐빗 위치별 확률 분포 리스트
            gate_name: 게이트 이름
            num_qubits_required: 필요한 큐빗 수
            available_qubits: 사용 가능한 총 큐빗 수
            temperature: 샘플링 온도
            
        Returns:
            선택된 큐빗 인덱스들 (요구사항 1: [n], [n,n], [n,n,n] 형태)
        """
        selected_qubits = []
        used_qubits = set()
        
        for i in range(min(num_qubits_required, len(qubit_probs))):
            probs = qubit_probs[i].clone()
            
            # 온도 스케일링
            if temperature != 1.0:
                probs = probs / temperature
                probs = F.softmax(probs, dim=-1)
            
            # 이미 사용된 큐빗과 "no qubit" 토큰 마스킹
            for used_qubit in used_qubits:
                if used_qubit < probs.shape[-1] - 1:  # -1은 "no qubit" 토큰
                    probs[0, used_qubit] = 0.0
            
            # 사용 가능한 큐빗 범위 밖은 마스킹
            if available_qubits < probs.shape[-1] - 1:
                probs[0, available_qubits:-1] = 0.0
            
            # 확률 재정규화
            probs = probs / probs.sum()
            
            # 큐빗 샘플링
            qubit_idx = torch.multinomial(probs.squeeze(0), 1)
            
            # "no qubit" 토큰이 아닌 경우에만 추가
            if qubit_idx < available_qubits:
                selected_qubits.append(qubit_idx)
                used_qubits.add(qubit_idx)
        
        return selected_qubits
    
    def sample_parameters_for_gate(
        self,
        param_values: List[torch.Tensor],
        gate_name: str,
        num_params_required: int
    ) -> List[float]:
        """게이트에 필요한 파라미터들을 추출
        
        Args:
            param_values: 예측된 파라미터 값들
            gate_name: 게이트 이름  
            num_params_required: 필요한 파라미터 수
            
        Returns:
            파라미터 값들 (요구사항 2)
        """
        parameters = []
        
        for i in range(min(num_params_required, len(param_values))):
            param_val = param_values[i].squeeze()
            
            # 파라미터 범위 제한 (회전 게이트의 경우 0 ~ 2π)
            if gate_name.startswith('r') or gate_name in ['p']:  # rx, ry, rz, p 게이트
                param_val = param_val % (2 * math.pi)
            
            parameters.append(param_val)
        
        return parameters

    def generate_autoregressive(
        self,
        prompt_tokens: List[int] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        reward_calculator=None,
        target_properties: Dict[str, float] = None,
        num_qubits: int = 4
    ) -> List[Dict]:
        """
        Property-guided Autoregressive 생성 (Decision Transformer 기반)
        
        Args:
            prompt_tokens: 초기 토큰 시퀀스 (선택적)
            max_length: 최대 생성 길이
            temperature: 샘플링 온도
            top_k: Top-k 샘플링
            reward_calculator: 보상 계산기
            target_properties: 목표 속성
            num_qubits: 큐빗 수
            
        Returns:
            생성된 게이트 시퀀스 [{'gate': str, 'qubits': List[int], 'params': List[float]}]
        """
        self.eval()
        
        # 초기 시퀀스 설정
        generated_gates = []
        current_context = {
            'states': [],
            'actions': [],
            'rewards': []
        }
        
        # 보상 가이던스 설정
        use_reward_guidance = reward_calculator is not None and target_properties is not None
        if use_reward_guidance:
            reward_calculator.set_target_properties(target_properties)
            print(f"🎯 Using reward guidance with targets: {target_properties}")
        
        with torch.no_grad():
            for step in range(max_length):
                print(f"\n--- Generation Step {step} ---")
                
                # 1. 현재 컨텍스트를 임베딩으로 변환
                if len(current_context['states']) == 0:
                    # 초기 상태 (빈 회로)
                    device = torch.device(self.device)
                    state_emb = torch.zeros(1, 1, self.d_model, device=device)
                    action_emb = torch.zeros(1, 1, self.d_model, device=device)
                    reward_emb = torch.zeros(1, 1, self.d_model, device=device)
                else:
                    # 기존 컨텍스트 재임베딩
                    state_emb = torch.stack(current_context['states'], dim=1)  # [1, seq_len, d_model]
                    action_emb = torch.stack(current_context['actions'], dim=1)
                    reward_emb = torch.stack(current_context['rewards'], dim=1)
                
                # 2. SAR 시퀀스 구성은 배치 콜레이터에서 처리됨
                # 여기서는 단순히 forward pass만 수행
                
                # 3. 어텐션 마스크 생성
                device = torch.device(self.device)
                mask = torch.ones(1, seq_len * 3, device=device, dtype=torch.bool)
                
                # 4. 트랜스포머 forward
                hidden_states = sar_sequence
                for block in self.transformer_blocks:
                    hidden_states = block(hidden_states, mask)
                
                # 5. 현재 상태 임베딩 추출 (마지막 state 위치)
                if seq_len > 0:
                    current_state_emb = hidden_states[:, (seq_len-1)*3, :]  # 마지막 state
                else:
                    current_state_emb = hidden_states[:, 0, :]  # 첫 번째 위치
                
                # 6. 보상 계산 (선택적)
                current_reward = 0.0
                if use_reward_guidance:
                    try:
                        reward_info = reward_calculator.calculate_reward_from_state_embedding(
                            current_state_emb, num_qubits=num_qubits
                        )
                        current_reward = reward_info['total_reward']
                        print(f"   Current reward: {current_reward:.4f}")
                        print(f"   Predicted properties: {reward_info['predicted_properties']}")
                        
                        # 높은 보상 달성 시 조기 종료
                        if current_reward > 0.8:
                            print(f"🎉 High reward achieved ({current_reward:.4f}), stopping generation")
                            break
                            
                    except Exception as e:
                        print(f"   Reward calculation failed: {e}")
                        current_reward = 0.0
                
                # 7. 다음 액션 예측
                gate_logits = self.gate_head(current_state_emb)  # [1, num_gates]
                
                # 보상 기반 바이어스 적용
                if use_reward_guidance and current_reward > 0:
                    reward_bias = current_reward * 2.0  # 보상 스케일링
                    gate_logits = gate_logits + reward_bias
                
                # 온도 스케일링 및 샘플링
                gate_probs = F.softmax(gate_logits / temperature, dim=-1)
                
                # Top-k 샘플링
                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(gate_probs, min(top_k, gate_probs.shape[-1]))
                    gate_idx = top_k_indices[0, torch.multinomial(top_k_probs[0], 1)]
                else:
                    gate_idx = torch.multinomial(gate_probs[0], 1)
                
                # 8. 게이트 정보 추출
                gate_registry = QuantumGateRegistry()
                gate_name = gate_registry.get_gate_name_by_index(gate_idx)
                
                if gate_name is None:
                    print(f"   Invalid gate index: {gate_idx}, stopping generation")
                    break
                
                print(f"   Selected gate: {gate_name} (idx: {gate_idx})")
                
                # 9. 큐빗 위치 예측
                position_logits = self.position_head(current_state_emb)  # [1, max_qubits]
                selected_qubits = self.sample_qubits_for_gate(
                    position_logits, gate_name, num_qubits, temperature
                )
                
                # 10. 파라미터 예측 (필요한 경우)
                param_values = []
                gate_info = gate_registry.get_gate_info(gate_name)
                if gate_info and gate_info.get('num_params', 0) > 0:
                    for i in range(gate_info['num_params']):
                        param_logit = self.parameter_heads[i](current_state_emb)
                        param_values.append(param_logit)
                    
                    parameters = self.sample_parameters_for_gate(
                        param_values, gate_name, gate_info['num_params']
                    )
                else:
                    parameters = []
                
                # 11. 생성된 게이트 저장
                generated_gate = {
                    'gate': gate_name,
                    'qubits': selected_qubits,
                    'params': parameters
                }
                generated_gates.append(generated_gate)
                
                print(f"   Generated: {generated_gate}")
                
                # 12. 컨텍스트 업데이트 (다음 스텝을 위해)
                # 예측된 게이트를 현재 회로에 추가하여 새로운 상태 임베딩 생성
                predicted_gate_info = {
                    'gate_name': gate_name,
                    'qubits': selected_qubits,
                    'parameter_value': parameters[0] if parameters else 0.0
                }
                
                # 현재까지의 게이트 리스트 구성
                current_circuit_gates = []
                for prev_gate in generated_gates[:-1]:  # 방금 추가한 게이트 제외
                    current_circuit_gates.append({
                        'gate_name': prev_gate['gate'],
                        'qubits': prev_gate['qubits'],
                        'parameter_value': prev_gate['params'][0] if prev_gate['params'] else 0.0
                    })
                
                # 새로운 통합 Facade를 사용한 임베딩 생성
                from encoding.unified_embedding_facade import UnifiedEmbeddingFacade
                facade = UnifiedEmbeddingFacade()
                
                # 현재 회로 상태를 grid matrix 형태로 변환
                grid_data = self._convert_circuit_to_grid_data(current_circuit_gates, num_qubits)
                
                # Facade를 통해 임베딩 생성
                embedding_result = facade.process_for_decision_transformer(
                    grid_data, 
                    original_gate_count=len(current_circuit_gates)
                )
                
                # 마지막 상태 임베딩 추출
                new_state_emb = embedding_result['input_sequence'][-1]
                
                # 액션 임베딩 (예측된 게이트를 위한 간단한 임베딩)
                device = torch.device(self.device)
                gate_tensor = torch.tensor([gate_idx, selected_qubits[0], 
                                          selected_qubits[1] if len(selected_qubits) > 1 else selected_qubits[0],
                                          parameters[0] if parameters else 0.0], 
                                         dtype=torch.float32, device=device)
                action_emb = torch.nn.functional.embedding(torch.tensor([gate_idx], device=device), 
                                                         torch.randn(self.n_gate_types, self.d_model, device=device))
                
                # 보상 임베딩 (간단한 선형 변환)
                reward_tensor = torch.tensor([current_reward], device=device)
                reward_emb_new = torch.nn.Linear(1, self.d_model).to(device)(reward_tensor.unsqueeze(0)).squeeze(0)
                
                # 컨텍스트에 추가
                current_context['states'].append(new_state_emb)
                current_context['actions'].append(action_emb)
                current_context['rewards'].append(reward_emb_new)
                
                # 종료 조건 체크
                if gate_name in ['measure', 'barrier'] or len(generated_gates) >= max_length:
                    break
        
        print(f"\n🎯 Generation completed: {len(generated_gates)} gates generated")
        return generated_gates


class DebugMode:
    """디버그 모드 설정"""
    TENSOR_DIM = "tensor_dim"          # 텐서 차원 테스트
    EMBEDDING = "embedding"            # 임베딩 디버그
    MODEL_PREDICTION = "model_prediction"  # 모델 예측 디버그
    MODEL_OUTPUT = "model_output"      # 모델 출력 디버그
    
    # 현재 활성화된 디버그 모드들
    ACTIVE_MODES = {MODEL_OUTPUT}
    
    @staticmethod
    def is_active(mode: str) -> bool:
        """디버그 모드가 활성화되어 있는지 확인"""
        return mode in DebugMode.ACTIVE_MODES


class ActionLossComputer:
    """액션 예측 손실 계산기 - 모듈러 구조에 맞게 리팩토링된 손실 계산 전용 클래스"""
    
    def __init__(self, loss_weights: Dict[str, float] = None, ignore_index: int = -100):
        self.weights = loss_weights or {'gate': 0.8, 'position': 0.1, 'parameter': 0.1}
        self.ignore_index = ignore_index
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=ignore_index)
        self.mse_loss = nn.MSELoss()
    
    def compute(self, predictions: Dict, targets: Dict, mask: torch.Tensor, num_qubits: Optional[List[int]] = None, num_gates: Optional[List[int]] = None) -> Dict:
        """통합 손실 계산 (동적 큐빗 마스킹 지원)"""
        device = mask.device
        
        #  DEBUG: 액션 마스크 차원 분석
        # Remove debug prints that access tensor values during training
        # Remove debug prints that access tensor values during training
        
        # 유효한 예측 위치만 선택
        valid_mask = mask.view(-1)
        # Remove debug prints that access tensor values during training
        # Remove debug prints that access tensor values during training
        
        if valid_mask.sum() == 0:
            return self._empty_loss_dict(device)
        
        # 동적 큐빗 마스킹 적용
        if num_qubits is not None and 'position_preds' in predictions:
            predictions = self._apply_dynamic_qubit_masking(predictions, num_qubits, mask)
        
        #  DEBUG: 손실 계산 진행 상황 추적
        # Remove debug prints that access tensor values during training
        # Remove debug prints that access tensor values during training
        
        # 각 손실 계산
        # Gate loss computation
        gate_loss = self._compute_gate_loss(predictions, targets, valid_mask)
        # Gate loss computed
        
        # Position loss computation
        position_loss = self._compute_position_loss(predictions, targets, valid_mask, num_qubits, num_gates)
        # Position loss computed
        
        # Parameter loss computation
        parameter_loss = self._compute_parameter_loss(predictions, targets, valid_mask)
        # Parameter loss computed
        
        #  DEBUG: 첫 번째 회로의 예측 vs 정답 비교
        #self._debug_first_circuit_predictions(predictions, targets, mask, num_gates)
        
        # 가중 합계
        total_loss = (
            self.weights['gate'] * gate_loss + 
            self.weights['position'] * position_loss + 
            self.weights['parameter'] * parameter_loss
        )
        
        # Remove debug prints that access tensor values during training
        
        losses = {
            'loss': total_loss,
            'gate_loss': gate_loss,
            'position_loss': position_loss,
            'parameter_loss': parameter_loss,
        }
        
        # Loss dictionary created
        
        # 정확도 및 분류 메트릭 추가 (하위 호환성)
        if hasattr(self, '_gate_accuracy'):
            losses['gate_accuracy'] = self._gate_accuracy
        if hasattr(self, '_gate_precision'):
            losses['gate_precision'] = self._gate_precision
        if hasattr(self, '_gate_recall'):
            losses['gate_recall'] = self._gate_recall
        if hasattr(self, '_gate_f1'):
            losses['gate_f1'] = self._gate_f1
        
        # Loss computation completed
        return losses
    
    def _debug_first_circuit_predictions(self, predictions: Dict, targets: Dict, mask: torch.Tensor, num_gates: Optional[List[int]] = None) -> None:
        """첫 번째 회로의 예측 vs 정답 비교 디버그"""
        try:
            print(f" [DEBUG_ENTRY] num_gates: {num_gates}")
            print(f" [DEBUG_ENTRY] mask.shape: {mask.shape}")
            print(f" [DEBUG_ENTRY] predictions keys: {list(predictions.keys())}")
            print(f" [DEBUG_ENTRY] targets keys: {list(targets.keys())}")
            
            if num_gates is None or len(num_gates) == 0:
                print(" [DEBUG_EXIT] num_gates가 None이거나 비어있음")
                return
            
            batch_size, seq_len = mask.shape
            first_circuit_gates = num_gates[0]
            print(f" [DEBUG_INFO] first_circuit_gates: {first_circuit_gates}")
            
            # 첫 번째 회로의 유효한 게이트 위치 찾기
            first_circuit_mask = mask[0]  # [seq_len]
            valid_positions = torch.where(first_circuit_mask)[0][:first_circuit_gates]  # 첫 N개 위치만
            print(f" [DEBUG_INFO] valid_positions: {valid_positions}")
            
            if len(valid_positions) == 0:
                print(" [DEBUG_EXIT] valid_positions가 비어있음")
                return
            
            print(f"\n🔍 [DEBUG] 첫 번째 회로 예측 분석 (게이트 수: {first_circuit_gates})")
            print("=" * 80)
        
            # Gate predictions vs targets
            if 'gate_logits' in predictions and 'gate_targets' in targets:
                gate_logits = predictions['gate_logits'][0]  # [seq_len, 20]
                gate_preds = torch.argmax(gate_logits, dim=-1)  # [seq_len]
                
                # 타겟 처리
                gate_targets = targets['gate_targets']
                if gate_targets.dim() == 1:
                    # [batch*seq] 형태인 경우
                    first_circuit_targets = gate_targets[:first_circuit_gates]
                else:
                    # [batch, seq] 형태인 경우
                    first_circuit_targets = gate_targets[0, :first_circuit_gates]
                
                print(f"🎯 Gate 예측 vs 정답 (처음 {min(10, first_circuit_gates)}개):")
                for i, pos in enumerate(valid_positions[:10]):
                    pred_gate = gate_preds[pos]
                    true_gate = first_circuit_targets[i] if i < len(first_circuit_targets) else -1
                    match = "✅" if pred_gate == true_gate else "❌"
                    print(f"   위치 {pos:2d}: 예측={pred_gate:2d}, 정답={true_gate:2d} {match}")
            
            # Position predictions vs targets  
            if 'position_preds' in predictions and 'position_targets' in targets:
                position_preds = predictions['position_preds'][0]  # [seq_len, 32, 2]
                position_logits = position_preds[:, 0, :]  # [seq_len, 2] - 첫 번째 큐빗만
                
                position_targets = targets['position_targets']
                if position_targets.dim() == 2 and position_targets.shape[1] == 2:
                    # [N, 2] 형태
                    first_circuit_pos_targets = position_targets[:first_circuit_gates]
                else:
                    print("   Position targets 형태를 파싱할 수 없음")
                    first_circuit_pos_targets = None
                
                if first_circuit_pos_targets is not None:
                    print(f"🎯 Position 예측 vs 정답 (처음 {min(5, first_circuit_gates)}개):")
                    for i, pos in enumerate(valid_positions[:5]):
                        pred_pos = position_logits[pos]  # [2]
                        true_pos = first_circuit_pos_targets[i] if i < len(first_circuit_pos_targets) else torch.tensor([-1, -1])
                        print(f"   위치 {pos:2d}: 예측=[{pred_pos[0]:.2f}, {pred_pos[1]:.2f}], 정답=[{true_pos[0]}, {true_pos[1]}]")
            
                print("=" * 80)
        
        except Exception as e:
            print(f" [DEBUG_ERROR] 디버그 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    def _compute_gate_loss(self, predictions, targets, valid_mask):
        """게이트 타입 손실 계산"""
        # Gate prediction logits
        gate_logits = predictions['gate_logits']  # [batch, seq, num_types]
        batch_size, seq_len, num_gate_types = gate_logits.shape
        
        # 로짓 재구성 (Reshape logits)
        reshaped_logits = gate_logits.reshape(-1, num_gate_types)  # [batch*seq, num_types]
        
        # 디버깅 정보
        # Remove debug prints that access tensor values during training
        
        # 마스크가 True인 위치만 선택 (직접 인덱싱)
        selected_indices = torch.where(valid_mask)[0]
        # Remove debug prints that access tensor values during training
        
        # 가능한 범위 내의 인덱스만 사용
        max_idx = reshaped_logits.shape[0] - 1
        valid_indices = selected_indices[selected_indices <= max_idx]
        
        # 마스크 적용 - 유효한 인덱스만 사용
        gate_logits = reshaped_logits[valid_indices]
        # Remove debug prints that access tensor values during training
        
        # 타겟 인덱싱 - 같은 방식으로 처리
        gate_targets = targets['gate_targets'].reshape(-1)
        
        # 🔥 CRITICAL FIX: Map special tokens to valid prediction range
        # Embedding uses 22 gates (0-21), but prediction head uses 20 gates (0-19)
        # Map special tokens: [EOS]=20 -> 0 (H gate), [PAD]=21 -> 0 (H gate)
        special_token_mask = gate_targets >= 20  # >= 20 for special tokens
        if special_token_mask.any():
            # Special token mapping to H gate
            gate_targets = gate_targets.clone()
            gate_targets[special_token_mask] = 0  # Map special tokens to H gate
        
        # 가능한 범위 내의 타겟만 사용
        if len(gate_targets) > len(valid_indices):
            gate_targets = gate_targets[valid_indices]
        else:
            # 타겟 배열이 더 작은 경우
            max_target_idx = min(len(gate_targets)-1, max(valid_indices))
            usable_indices = valid_indices[valid_indices <= max_target_idx]
            gate_logits = reshaped_logits[usable_indices]
            gate_targets = gate_targets[usable_indices]
            pass  # Target size limitation warning removed
        
        # Remove debug prints that access tensor values during training
        
        # 텐서 타입을 Long으로 변환 (cross_entropy는 Long 타입을 요구함)
        if gate_targets.dtype == torch.bool:
            gate_targets = gate_targets.long()
        elif gate_targets.dtype != torch.long:
            gate_targets = gate_targets.to(torch.long)
        
        # 패딩 타겟(-1) 필터링
        valid_target_mask = gate_targets >= 0
        if valid_target_mask.sum() == 0:
            return torch.tensor(0.0, device=gate_logits.device)
        
        final_logits = gate_logits[valid_target_mask]
        final_targets = gate_targets[valid_target_mask]
        
        # 정확도 계산 및 저장
        gate_predictions = torch.argmax(final_logits, dim=-1)
        self._gate_accuracy = (gate_predictions == final_targets).float().mean()
        
        # F1, Precision, Recall 계산 (다중 클래스)
        self._compute_classification_metrics(gate_predictions, final_targets)
        
        return self.focal_loss(final_logits, final_targets)
    
    def _compute_classification_metrics(self, predictions: torch.Tensor, targets: torch.Tensor):
        """F1, Precision, Recall 계산 (매크로 평균)"""
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # 고유 클래스들
        unique_classes = np.unique(np.concatenate([predictions_np, targets_np]))
        
        if len(unique_classes) <= 1:
            # 단일 클래스인 경우
            self._gate_precision = 1.0
            self._gate_recall = 1.0
            self._gate_f1 = 1.0
            return
        
        # 클래스별 precision, recall 계산
        precisions = []
        recalls = []
        f1s = []
        
        for cls in unique_classes:
            # True Positive, False Positive, False Negative
            tp = np.sum((predictions_np == cls) & (targets_np == cls))
            fp = np.sum((predictions_np == cls) & (targets_np != cls))
            fn = np.sum((predictions_np != cls) & (targets_np == cls))
            
            # Precision, Recall 계산
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        # 매크로 평균
        self._gate_precision = np.mean(precisions)
        self._gate_recall = np.mean(recalls)
        self._gate_f1 = np.mean(f1s)
    
    def _compute_position_loss(self, predictions: Dict, targets: Dict, valid_mask: torch.Tensor,
                               num_qubits: Optional[List[int]] = None, num_gates: Optional[List[int]] = None) -> torch.Tensor:
        position_preds = predictions['position_preds']  # [batch, seq_len, max_qubits, 2]
        batch_size, seq_len, max_qubits, pos_dim = position_preds.shape
        
        # Remove debug prints that access tensor values during training
        
        # 첫 번째 큐빗 위치만 예측 (간단화)
        position_logits = position_preds[:, :, 0, :]  # [batch, seq_len, 2]
        position_logits_flat = position_logits.reshape(-1, 2)  # [batch*seq_len, 2]
        
        # valid_mask로 실제 게이트 위치만 선택
        valid_position_logits = position_logits_flat[valid_mask]  # [num_valid_gates, 2]
        # Remove debug prints that access tensor values during training
        
        # position_targets 가져오기 (qubit_targets로 매핑)
        if 'qubit_targets' not in targets:
            # Missing qubit_targets key
            return torch.tensor(0.0, device=position_preds.device)
        
        position_targets = targets['qubit_targets']  # [batch, seq_len, 2]
        
        # 🔥 FIX: Handle case where position_targets is a list instead of tensor
        if isinstance(position_targets, list):
            # Converting position_targets list to tensor
            # Convert list to tensor, handle nested lists
            if len(position_targets) > 0 and isinstance(position_targets[0], list):
                # Nested list case: [[qubits], [qubits], ...]
                max_len = max(len(item) if isinstance(item, list) else 1 for item in position_targets)
                padded_targets = []
                for item in position_targets:
                    if isinstance(item, list):
                        # Pad to max_len with -1
                        padded = item + [-1] * (max_len - len(item))
                        padded_targets.append(padded[:2])  # Take first 2 qubits only
                    else:
                        padded_targets.append([item, -1])  # Single qubit case
                position_targets = torch.tensor(padded_targets, dtype=torch.long)
            else:
                # Simple list case
                position_targets = torch.tensor(position_targets, dtype=torch.long)
        
        # Remove debug prints that access tensor values during training
        
        # position_targets를 flat하게 변환
        position_targets_flat = position_targets.reshape(-1, 2)  # [batch*seq_len, 2]
        # Remove debug prints that access tensor values during training
        
        # valid_mask로 필터링
        valid_position_targets = position_targets_flat[valid_mask]
        # Remove debug prints that access tensor values during training
        
        # 패딩된 타겟(-1) 제거
        non_padding_mask = (valid_position_targets[:, 0] >= 0) & (valid_position_targets[:, 1] >= 0)
        # Remove debug prints that access tensor values during training
        
        if non_padding_mask.sum() == 0:
            # All position targets are padding (-1)
            return torch.tensor(0.0, device=position_preds.device)
        
        final_preds = valid_position_logits[non_padding_mask]
        final_targets = valid_position_targets[non_padding_mask].float()
        
        # 🚨 강력한 큐빗 범위 페널티 적용
        position_loss = F.mse_loss(final_preds, final_targets, reduction='mean')
        
        # 큐빗 범위 위반 페널티 계산
        if num_qubits is not None:
            penalty_loss = self._compute_qubit_range_penalty(
                final_preds, final_targets, num_qubits, valid_mask, non_padding_mask
            )
            # 페널티를 기본 손실에 추가 (강력한 가중치 적용)
            position_loss = position_loss + 10.0 * penalty_loss
        
        return position_loss
    
    def _compute_qubit_range_penalty(self, predictions: torch.Tensor, targets: torch.Tensor, 
                                   num_qubits: List[int], valid_mask: torch.Tensor, 
                                   non_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        큐빗 범위를 벗어나는 예측에 대한 강력한 페널티 계산
        
        Args:
            predictions: 예측된 큐빗 위치 [num_valid, 2]
            targets: 실제 큐빗 위치 [num_valid, 2]  
            num_qubits: 각 회로의 큐빗 수 [batch_size]
            valid_mask: 유효한 위치 마스크
            non_padding_mask: 패딩이 아닌 위치 마스크
        
        Returns:
            페널티 손실 (큐빗 범위 위반시 큰 값)
        """
        device = predictions.device
        penalty_loss = torch.tensor(0.0, device=device)
        
        # 배치별로 큐빗 범위 확인
        batch_size = len(num_qubits)
        
        # valid_mask와 non_padding_mask를 통해 배치 인덱스 복원
        batch_indices = []
        current_idx = 0
        
        for batch_idx in range(batch_size):
            # 이 배치의 유효한 위치 수 계산 (근사치)
            batch_valid_count = valid_mask.sum() // batch_size
            
            for _ in range(batch_valid_count):
                if current_idx < len(predictions):
                    batch_indices.append(batch_idx)
                    current_idx += 1
        
        # 예측값이 큐빗 범위를 벗어나는지 확인
        total_violations = 0
        total_penalty = 0.0
        
        for i, pred in enumerate(predictions):
            if i < len(batch_indices):
                batch_idx = batch_indices[i]
                max_qubit = num_qubits[batch_idx] - 1  # 0-indexed
                
                # 두 큐빗 위치 모두 확인
                qubit1, qubit2 = pred[0], pred[1]
                
                # 범위 위반 검사
                violation_penalty = 0.0
                
                # 큐빗1 범위 위반
                if qubit1 < 0 or qubit1 > max_qubit:
                    violation_penalty += torch.abs(qubit1 - torch.clamp(qubit1, 0, max_qubit))
                    total_violations += 1
                
                # 큐빗2 범위 위반  
                if qubit2 < 0 or qubit2 > max_qubit:
                    violation_penalty += torch.abs(qubit2 - torch.clamp(qubit2, 0, max_qubit))
                    total_violations += 1
                
                total_penalty += violation_penalty
        
        # 위반이 있으면 강력한 페널티 적용
        if total_violations > 0:
            penalty_loss = torch.tensor(total_penalty, device=device)
            # Qubit range violation penalty applied
        
        return penalty_loss
    
    def _compute_parameter_loss(self, predictions: Dict, targets: Dict, valid_mask: torch.Tensor) -> torch.Tensor:
        # 파라미터 예측 정보 차원 디버깅
        param_preds = predictions['parameter_preds']
        debug_print(f" PARAMETER_PREDS_SHAPE: {param_preds.shape}")
        
        # 리서이핑
        reshaped_preds = param_preds.reshape(-1)
        debug_print(f" RESHAPED_PARAM_PREDS_SHAPE: {reshaped_preds.shape}")
        
        # 안전하게 마스크 적용 (차원이 맞지 않으면 예외 발생)
        if len(reshaped_preds) != len(valid_mask):
            debug_print(f"⚠️ 파라미터 차원 불일치 발생! 가장 작은 차원으로 잘라냄")
            min_len = min(len(reshaped_preds), len(valid_mask))
            valid_mask = valid_mask[:min_len]
            reshaped_preds = reshaped_preds[:min_len]
        
        # 마스크 적용
        param_preds = reshaped_preds[valid_mask]
        
        # parameter_targets 확인
        if 'parameter_targets' not in targets:
            return torch.tensor(0.0, device=param_preds.device)
        
        # parameter_targets가 리스트인 경우 텐서로 변환
        if isinstance(targets['parameter_targets'], list):
            if len(targets['parameter_targets']) == 0:
                return torch.tensor(0.0, device=param_preds.device)
            parameter_targets_tensor = torch.tensor(targets['parameter_targets'], device=param_preds.device)
        else:
            parameter_targets_tensor = targets['parameter_targets']
        
        # 타겟 재생성
        reshaped_targets = parameter_targets_tensor.reshape(-1)
        if len(reshaped_targets) > len(valid_mask):
            reshaped_targets = reshaped_targets[:len(valid_mask)]
        param_targets = reshaped_targets[valid_mask]
        
        # NaN 처리
        non_nan_mask = ~torch.isnan(param_targets)
        if non_nan_mask.sum() > 0:
            return self.mse_loss(param_preds[non_nan_mask], param_targets[non_nan_mask])
        return torch.tensor(0.0, device=param_preds.device)
    
    def _apply_dynamic_qubit_masking(self, predictions: Dict, num_qubits: List[int], mask: torch.Tensor) -> Dict:
        """🚀 엄격한 동적 큐빗 마스킹: 각 회로의 정확한 큐빗 수에 맞게 예측을 마스킹"""
        if 'position_preds' not in predictions:
            raise ValueError("❌ CRITICAL ERROR: position_preds가 예측에 포함되지 않았습니다!")
        
        if num_qubits is None:
            raise ValueError("❌ CRITICAL ERROR: num_qubits 정보가 누락되었습니다!")
        
        # 예측 복사 (원본 수정 방지)
        masked_predictions = predictions.copy()
        position_preds = predictions['position_preds'].clone()  # [batch, num_actions, max_qubits, 2]
        
        batch_size, num_actions, max_qubits, qubit_dims = position_preds.shape
        
        # 🚀 배치 크기 검증
        if len(num_qubits) != batch_size:
            raise ValueError(
                f"❌ CRITICAL ERROR: 배치 크기 불일치!\n"
                f"   position_preds 배치 크기: {batch_size}\n"
                f"   num_qubits 길이: {len(num_qubits)}"
            )
        
        # 🚀 각 배치별로 정확한 동적 마스킹 적용
        for batch_idx in range(batch_size):
            circuit_qubits = num_qubits[batch_idx]
            
            # 🚀 큐빗 수 검증 및 자동 조정
            if circuit_qubits <= 0:
                raise ValueError(f"❌ CRITICAL ERROR: 회로 {batch_idx}의 큐빗 수가 0 이하입니다: {circuit_qubits}")
            
            if circuit_qubits > max_qubits:
                debug_print(f"⚠️ 회로 {batch_idx}: 큐빗 수 초과 감지 - 자동 조정")
                debug_print(f"   원본 큐빗 수: {circuit_qubits}")
                debug_print(f"   모델 최대 큐빗: {max_qubits}")
                debug_print(f"   → {max_qubits}개로 제한")
                circuit_qubits = max_qubits
            
            # ✅ 유효하지 않은 큐빗 인덱스를 -inf로 마스킹 (softmax에서 확률 0이 됨)
            for action_idx in range(num_actions):
                for qubit_dim in range(qubit_dims):  # qubit1, qubit2
                    # circuit_qubits 이상의 인덱스는 -inf로 마스킹
                    position_preds[batch_idx, action_idx, circuit_qubits:, qubit_dim] = float('-inf')
        
        masked_predictions['position_preds'] = position_preds
        return masked_predictions
    
    def _empty_loss_dict(self, device: torch.device) -> Dict:
        return {
            'total_loss': torch.tensor(0.0, device=device),
            'gate_loss': torch.tensor(0.0, device=device),
            'position_loss': torch.tensor(0.0, device=device),
            'parameter_loss': torch.tensor(0.0, device=device),
            'gate_accuracy': torch.tensor(0.0, device=device)
        }

    def forward(self, predictions, targets, action_prediction_mask, eos_mask=None, num_qubits=None):
        """ 클린한 손실 계산 (복잡한 로직 제거)"""
        device = action_prediction_mask.device
        
        # 새로운 타겟 구조 처리
        if 'action_targets' in targets and targets['action_targets']:
            action_targets = targets['action_targets']
            
            # ActionLossComputer 사용
            loss_computer = ActionLossComputer()
            combined_mask = action_prediction_mask & eos_mask if eos_mask is not None else action_prediction_mask
            
            return loss_computer.compute(predictions, action_targets, combined_mask, num_qubits=num_qubits)
        
        # 🔥 NEW: legacy 타겟 구조를 ActionLossComputer로 변환
        if 'target_actions' in targets:
            # legacy 타겟을 새로운 구조로 변환
            action_targets = {
                'gate_targets': targets['target_actions'],
                'position_targets': targets.get('target_qubits'),
                'parameter_targets': targets.get('target_params')
            }
            
            # ActionLossComputer 사용
            loss_computer = ActionLossComputer()
            combined_mask = action_prediction_mask & eos_mask if eos_mask is not None else action_prediction_mask
            
            return loss_computer.compute(predictions, action_targets, combined_mask, num_qubits=num_qubits)
        
        # FALLBACK: 기존 복잡한 로직 (하위 호환성)
        return self._legacy_loss_computation(predictions, targets, action_prediction_mask, eos_mask, num_qubits)
    
    def _legacy_loss_computation(self, predictions, targets, action_prediction_mask, eos_mask, num_qubits):
        """기존 복잡한 손실 계산 로직 (하위 호환성)"""
        device = action_prediction_mask.device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}
        
        # EOS 토큰 이후 위치 마스킹
        eos_mask = self.create_eos_mask(target_gates)
        
        # 디버깅: 마스크 크기 확인
        if DebugMode.is_active(DebugMode.MODEL_PREDICTION):
            print(f" MASK_DEBUG: action_prediction_mask.shape = {action_prediction_mask.shape}")
            print(f" MASK_DEBUG: eos_mask.shape = {eos_mask.shape}")
            print(f" MASK_DEBUG: target_gates.shape = {target_gates.shape if target_gates is not None else 'None'}")
        
        # 액션 예측 마스크와 EOS 마스크 결합 (크기 불일치 시 에러 발생)
        combined_mask = action_prediction_mask & eos_mask
        
        # 마스킹된 위치에서만 손실 계산
        mask_sum = combined_mask.sum()
        
        if mask_sum == 0:
            if DebugMode.is_active(DebugMode.MODEL_PREDICTION):
                print(" WARNING: No valid predictions to compute loss!")
            return {
                'total_loss': total_loss,
                'gate_loss': torch.tensor(0.0, device=device),
                'qubit_loss': torch.tensor(0.0, device=device),
                'param_loss': torch.tensor(0.0, device=device),
                'gate_accuracy': torch.tensor(0.0, device=device),
                'num_predictions': torch.tensor(0, device=device)
            }
        
        masked_gate_logits = gate_type_logits[combined_mask]
        masked_gate_targets = target_gates[combined_mask]
        
        # 모델 예측 디버그 (선택적 출력)
        if DebugMode.is_active(DebugMode.MODEL_PREDICTION):
            # 패딩 토큰(0번) 제외하고 실제 게이트만 분석
            non_pad_mask = masked_gate_targets != 0
            non_pad_targets = masked_gate_targets[non_pad_mask]
            
            if len(non_pad_targets) > 0:
                unique_targets = torch.unique(non_pad_targets)
                target_dist = torch.bincount(non_pad_targets)
                print(f" REAL_TARGETS: unique={unique_targets.tolist()}, count={len(unique_targets)}")
                print(f" REAL_DIST: {target_dist.tolist()}")
                # Padding ratio calculation removed to avoid tensor access
            else:
                print(" No non-padding targets found!")
            
            # 큐빗 위치 타겟 분석
            if target_qubits:
                for i, qubit_targets in enumerate(target_qubits):
                    if qubit_targets is not None:
                        masked_qubit_targets = qubit_targets[combined_mask]
                        valid_qubits = masked_qubit_targets[masked_qubit_targets >= 0]  # -1 제외
                        if len(valid_qubits) > 0:
                            unique_qubits = torch.unique(valid_qubits)
                            print(f" QUBIT_{i}: unique={unique_qubits.tolist()}, count={len(unique_qubits)}")
            
            # 파라미터 타겟 분석
            if target_params:
                for i, param_targets in enumerate(target_params):
                    if param_targets is not None:
                        masked_param_targets = param_targets[combined_mask]
                        valid_params = masked_param_targets[~torch.isnan(masked_param_targets)]  # NaN 제외
                        if len(valid_params) > 0:
                            # Parameter range calculation removed to avoid tensor access
                            print(f"🎯 PARAM_{i}: range={param_range}, count={len(valid_params)}")
        
        gate_loss = self.cross_entropy(masked_gate_logits, masked_gate_targets)
        total_loss = total_loss + self.gate_weight * gate_loss
        loss_dict['gate_loss'] = gate_loss
        
        with torch.no_grad():
            gate_predictions = torch.argmax(masked_gate_logits, dim=-1)
            gate_accuracy = (gate_predictions == masked_gate_targets).float().mean()
            
            # 모델 예측 디버그 (선택적 출력)
            if DebugMode.is_active(DebugMode.MODEL_PREDICTION):
                unique_preds = torch.unique(gate_predictions)
                print(f"🤖 PREDICTIONS: unique={unique_preds.tolist()}, count={len(unique_preds)}")
                print(f"📈 ACCURACY: gate={gate_accuracy:.4f}")
                
            loss_dict['gate_accuracy'] = gate_accuracy
        
        # 2. 큐빗 위치 손실 (요구사항 1: 통합 텐서로 처리)
        if target_qubits is not None:
            # 새로운 통합 텐서 구조 처리: [batch, num_gates, max_qubits_per_gate]
            if isinstance(target_qubits, torch.Tensor) and target_qubits.dim() == 3:
                # 통합 큐빗 예측: [num_actions, max_qubits_per_gate, max_qubits + 1]
                masked_qubit_logits = qubit_position_logits[combined_mask]
                
                # 통합 큐빗 타겟: [num_actions, max_qubits_per_gate]
                masked_qubit_targets = target_qubits[combined_mask]
                
                # 동적 큐빗 마스킹 적용
                if num_qubits is not None:
                    num_actions = masked_qubit_logits.shape[0]
                    max_qubits_per_gate = masked_qubit_logits.shape[1]
                    
                    for action_idx in range(num_actions):
                        # 각 회로의 실제 큐빗 수를 넘는 예측은 마스킹
                        circuit_idx = action_idx  # 배치 내 회로 인덱스 (단순화)
                        if circuit_idx < len(num_qubits):
                            max_valid_qubit = num_qubits[circuit_idx] if isinstance(num_qubits[circuit_idx], int) else num_qubits[circuit_idx].item()
                            # 유효하지 않은 큐빗 인덱스는 매우 작은 값으로 마스킹
                            masked_qubit_logits[action_idx, qubit_pos, max_valid_qubit+1:] = -1e9
                
                # 유효한 타겟이 있는 위치만 손실 계산
                valid_mask = masked_qubit_targets != self.ignore_index
                if valid_mask.sum() > 0:
                    # Flatten for cross entropy: [num_valid_predictions, max_qubits + 1]
                    valid_logits = masked_qubit_logits[valid_mask]  # [num_valid, max_qubits + 1]
                    valid_targets = masked_qubit_targets[valid_mask]  # [num_valid]
                    
                    qubit_loss = self.cross_entropy(valid_logits, valid_targets)
                    total_loss = total_loss + self.qubit_weight * qubit_loss
                    loss_dict['qubit_loss'] = qubit_loss
                else:
                    loss_dict['qubit_loss'] = torch.tensor(0.0, device=device)
            
            # 기존 리스트 구조 처리 (하위 호환성)
            elif isinstance(target_qubits, list) and len(target_qubits) > 0:
                # 통합 큐빗 예측: [num_actions, max_qubits_per_gate, max_qubits + 1]
                masked_qubit_logits = qubit_position_logits[combined_mask]
                
                # 타겟 큐빗들을 통합 텐서로 변환: [num_actions, max_qubits_per_gate]
                num_actions = masked_qubit_logits.shape[0]
                max_qubits_per_gate = masked_qubit_logits.shape[1]
                
                # 통합 타겟 텐서 생성
                unified_qubit_targets = torch.full(
                    (num_actions, max_qubits_per_gate), 
                    self.ignore_index, 
                    dtype=torch.long, 
                    device=device
                )
                
                # 각 큐빗 위치별 타겟을 통합 텐서에 복사
                for qubit_idx, qubit_targets in enumerate(target_qubits):
                    if qubit_targets is not None and qubit_idx < max_qubits_per_gate:
                        masked_targets = qubit_targets[combined_mask]
                        unified_qubit_targets[:, qubit_idx] = masked_targets
                
                # 동적 큐빗 마스킹 적용
                if num_qubits is not None:
                    for action_idx in range(num_actions):
                        for qubit_pos in range(max_qubits_per_gate):
                            # 각 회로의 실제 큐빗 수를 넘는 예측은 마스킹
                            circuit_idx = action_idx  # 배치 내 회로 인덱스 (단순화)
                            if circuit_idx < len(num_qubits):
                                max_valid_qubit = num_qubits[circuit_idx]
                                # 유효하지 않은 큐빗 인덱스는 매우 작은 값으로 마스킹
                                masked_qubit_logits[action_idx, qubit_pos, max_valid_qubit+1:] = -1e9
                
                # 유효한 타겟이 있는 위치만 손실 계산
                valid_mask = unified_qubit_targets != self.ignore_index
                if valid_mask.sum() > 0:
                    # Flatten for cross entropy: [num_valid_predictions, max_qubits + 1]
                    valid_logits = masked_qubit_logits[valid_mask]  # [num_valid, max_qubits + 1]
                    valid_targets = unified_qubit_targets[valid_mask]  # [num_valid]
                    
                    qubit_loss = self.cross_entropy(valid_logits, valid_targets)
                    total_loss = total_loss + self.qubit_weight * qubit_loss
                    loss_dict['qubit_loss'] = qubit_loss
                else:
                    loss_dict['qubit_loss'] = torch.tensor(0.0, device=device)
            else:
                loss_dict['qubit_loss'] = torch.tensor(0.0, device=device)
        else:
            loss_dict['qubit_loss'] = torch.tensor(0.0, device=device)
        
        # 3. 파라미터 손실 (요구사항 2: 통합 텐서로 처리)
        if target_params is not None:
            # 새로운 통합 텐서 구조 처리: [batch, num_gates, max_params_per_gate]
            if isinstance(target_params, torch.Tensor) and target_params.dim() == 3:
                # 통합 파라미터 예측: [num_actions, max_params_per_gate]
                masked_param_preds = parameter_predictions[combined_mask]
                
                # 통합 파라미터 타겟: [num_actions, max_params_per_gate]
                masked_param_targets = target_params[combined_mask]
                
                # 유효한 파라미터 타겟만 손실 계산 (NaN 제외)
                valid_mask = ~torch.isnan(masked_param_targets)
                non_padding_mask = position_targets != -1
        if non_padding_mask.sum() > 0:
            # 🚨 개별 회로 큐빗 수 기반 동적 정규화
            if num_qubits is not None and len(num_qubits) > 0:
                # 배치 내 각 샘플의 큐빗 수를 사용하여 정규화
                batch_size = len(num_qubits)
                seq_len = position_targets.size(0) // batch_size
                # 통합 파라미터 예측: [num_actions, max_params_per_gate]
                masked_param_preds = parameter_predictions[combined_mask]
                
                # 타겟 파라미터들을 통합 텐서로 변환: [num_actions, max_params_per_gate]
                num_actions = masked_param_preds.shape[0]
                max_params_per_gate = masked_param_preds.shape[1]
                
                # 통합 타겟 텐서 생성 (NaN으로 초기화)
                unified_param_targets = torch.full(
                    (num_actions, max_params_per_gate), 
                    float('nan'), 
                    dtype=torch.float32, 
                    device=device
                )
                
                # 각 파라미터별 타겟을 통합 텐서에 복사
                for param_idx, param_targets in enumerate(target_params):
                    if param_targets is not None and param_idx < max_params_per_gate:
                        masked_targets = param_targets[combined_mask]
                        unified_param_targets[:, param_idx] = masked_targets
                
                # 유효한 파라미터 타겟만 손실 계산 (NaN 제외)
                valid_mask = ~torch.isnan(unified_param_targets)
                if valid_mask.sum() > 0:
                    valid_preds = masked_param_preds[valid_mask]
                    valid_targets = unified_param_targets[valid_mask]
                    
                    param_loss = self.mse_loss(valid_preds, valid_targets)
                    total_loss = total_loss + self.param_weight * param_loss
                    loss_dict['param_loss'] = param_loss
                else:
                    loss_dict['param_loss'] = torch.tensor(0.0, device=device)
            else:
                loss_dict['param_loss'] = torch.tensor(0.0, device=device)
        else:
            loss_dict['param_loss'] = torch.tensor(0.0, device=device)
        
        # 🔥 CRITICAL: gate_accuracy가 항상 포함되도록 보장
        if 'gate_accuracy' not in loss_dict:
            loss_dict['gate_accuracy'] = torch.tensor(0.0, device=device)
        
        loss_dict.update({
            'total_loss': total_loss,
            'num_predictions': combined_mask.sum()
        })
        
        return loss_dict


# 모델 팩토리 함수
def create_decision_transformer(
    config = None,
    d_model: int = 512,
    n_layers: int = 6,
    n_heads: int = 8,
    n_gate_types: int = 20,
    dropout: float = 0.1,
    property_prediction_model: Optional[UnifiedPropertyPredictionTransformer] = None
) -> DecisionTransformer:
    """Decision Transformer 모델 생성"""
    
    # config가 딕셔너리인 경우 처리
    if config is not None:
        if isinstance(config, dict):
            d_model = config.get('d_model')
            n_layers = config.get('n_layers')
            n_heads = config.get('n_heads')
            dropout = config.get('dropout')
            attention_mode = config.get('attention_mode', 'standard')
            d_ff = config.get('d_ff', d_model * 4)
            property_prediction_model = config.get('property_prediction_model', property_prediction_model)
        else:
            # config 객체인 경우
            d_model = getattr(config, 'd_model')
            n_layers = getattr(config, 'n_layers')
            n_heads = getattr(config, 'n_heads')
            dropout = getattr(config, 'dropout')
            attention_mode = getattr(config, 'attention_mode', 'standard')
            d_ff = getattr(config, 'd_ff', d_model * 4)
    else:
        d_ff = d_model * 4  # 표준 비율
        attention_mode = 'standard'
    
    return DecisionTransformer(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        n_gate_types=n_gate_types,
        dropout=dropout,
        attention_mode=attention_mode,
        property_prediction_model=property_prediction_model
    )


# 사용 예시
if __name__ == "__main__":
    # 모델 생성
    model = create_decision_transformer(
        d_model=256,
        n_layers=4,
        n_heads=8,
        n_gate_types=20  # 🔧 FIXED: 통일된 게이트 타입 수
    )
    
    # 더미 데이터로 테스트
    batch_size, seq_len, d_model = 2, 10, 256
    
    input_sequence = torch.randn(batch_size, seq_len, d_model)
    attention_mask = torch.tril(torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool))
    action_prediction_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    action_prediction_mask[:, 1::3] = True  # 액션 위치
    
    # 순전파
    outputs = model(input_sequence, attention_mask, action_prediction_mask)
    
    # Remove debug prints that access tensor values during training
    
    # 🎯 손실 계산 테스트 (새로운 구조)
    # 더미 타겟 데이터 생성
    targets = {
        'gate_targets': torch.randint(0, 16, (batch_size, seq_len)),
        'position_targets': torch.randn(batch_size, seq_len, 3),
        'parameter_targets': torch.randn(batch_size, seq_len)
    }
    
    # 모델의 compute_loss 메서드 사용
    loss_outputs = model.compute_loss(
        predictions=outputs,
        targets=targets,
        action_prediction_mask=action_prediction_mask
    )
    
    # Loss output printing removed to avoid tensor access during training
