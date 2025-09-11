"""
완전 통합된 Property Prediction Transformer
레거시 의존성 없이 독립적으로 구현된 최종 버전
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
# 동적 임포트를 위한 경로 설정
import sys
from pathlib import Path

# 현재 파일의 부모 디렉토리들을 경로에 추가
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
project_root = src_dir.parent
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root / "quantumcommon"))

# UnifiedEmbeddingFacade 임포트
try:
    from encoding.unified_embedding_facade import UnifiedEmbeddingFacade
except ImportError:
    try:
        from ..encoding.unified_embedding_facade import UnifiedEmbeddingFacade
    except ImportError:
        from OAT_Model.src.encoding.unified_embedding_facade import UnifiedEmbeddingFacade

# OptimalPropertyHead 임포트
try:
    from models.optimal_property_prediction_head import OptimalPropertyHead, OptimalPropertyLoss
except ImportError:
    try:
        from .optimal_property_prediction_head import OptimalPropertyHead, OptimalPropertyLoss
    except ImportError:
        from OAT_Model.src.models.optimal_property_prediction_head import OptimalPropertyHead, OptimalPropertyLoss

# 게이트 레지스트리 임포트
try:
    from gates import QuantumGateRegistry
except ImportError:
    try:
        from quantumcommon.gates import QuantumGateRegistry
    except ImportError:
        print("Warning: Could not import QuantumGateRegistry")
        QuantumGateRegistry = None

# Import unified config instead of defining separately
try:
    from config.unified_training_config import PropertyConfig as UnifiedPropertyPredictionConfig
except ImportError:
    try:
        from ..config.unified_training_config import PropertyConfig as UnifiedPropertyPredictionConfig
    except ImportError:
        from OAT_Model.src.config.unified_training_config import PropertyConfig as UnifiedPropertyPredictionConfig


class TransformerBlock(nn.Module):
    """트랜스포머 블록"""
    def __init__(self, config: UnifiedPropertyPredictionConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class UnifiedPropertyPredictionTransformer(nn.Module):
    """완전 통합된 Property Prediction Transformer"""
    
    def __init__(self, config: UnifiedPropertyPredictionConfig):
        super().__init__()
        self.config = config
        
        # Circuit embedding with unified facade - pass config object
        embedding_config = {
            'd_model': config.d_model,
            'max_qubits': config.max_qubits,
            'max_gates': config.max_gates,
            'num_heads': config.n_heads,
            'dropout': config.dropout
        }
        self.circuit_embedding = UnifiedEmbeddingFacade(embedding_config)
        
        # 트랜스포머 레이어들
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # SOTA Prediction Head
        self.prediction_head = OptimalPropertyHead(
            d_model=config.d_model,
            dropout=config.dropout
        )
        
        # SOTA Loss Function
        self.loss_function = OptimalPropertyLoss(
            property_weights={
                'entanglement': 1.0,
                'fidelity': 1.0,
                'expressibility': 1.0
            }
        )
        
        # 초기화
        self._initialize_weights()
        
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _process_circuit_specs(self, circuit_specs):
        """CircuitSpec 객체들을 텐서 형태로 변환 - gate vocab 사용"""
        if not isinstance(circuit_specs, list):
            circuit_specs = [circuit_specs]
        
        batch_size = len(circuit_specs)
        max_gates = max(len(spec.gates) for spec in circuit_specs) if circuit_specs else 0
        max_gates = min(max_gates, self.config.max_gates)  # 최대 길이 제한
        
        # Gate registry에서 통일된 vocab 사용
        gate_registry = QuantumGateRegistry()
        gate_vocab = gate_registry.get_gate_vocab()
        
        # Get device from model parameters
        device = next(self.parameters()).device
        
        gate_types = torch.zeros(batch_size, max_gates, dtype=torch.long, device=device)
        qubit_indices = torch.zeros(batch_size, max_gates, 2, dtype=torch.long, device=device)
        parameters = torch.zeros(batch_size, max_gates, 3, device=device)
        
        for i, spec in enumerate(circuit_specs):
            for j, gate in enumerate(spec.gates[:max_gates]):
                # 게이트 타입 - unified vocab 사용
                gate_types[i, j] = gate_vocab.get(gate.name.lower(), 0)
                
                # 큐빗 인덱스
                if gate.qubits:
                    qubit_indices[i, j, :len(gate.qubits[:2])] = torch.tensor(gate.qubits[:2], device=device)
                
                # 파라미터
                if gate.parameters:
                    parameters[i, j, :len(gate.parameters[:3])] = torch.tensor(gate.parameters[:3], device=device)
        
        return {
            'gate_types': gate_types,
            'qubit_indices': qubit_indices,
            'parameters': parameters
        }

    def forward(self, circuit_spec, targets: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # CircuitSpec을 텐서 형태로 변환 (단일 객체인 경우 리스트로 변환)
        if not isinstance(circuit_spec, list):
            circuit_spec = [circuit_spec]
        circuit_data = self._process_circuit_specs(circuit_spec)
        
        # 회로 임베딩
        embedding_output = self.circuit_embedding(circuit_data)
        
        # Extract main embeddings tensor from dictionary
        embedded = embedding_output['embeddings'] if isinstance(embedding_output, dict) else embedding_output
        
        # 트랜스포머 레이어들 통과
        x = embedded
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 시퀀스 풀링 (평균)
        pooled = x.mean(dim=1)  # [batch_size, d_model]
        
        # 속성 예측 (targets 전달하여 러닝 통계 업데이트)
        # 추론 모드에서는 자동으로 denormalization 수행
        inference_mode = not self.training
        predictions = self.prediction_head(pooled, targets=targets, inference_mode=inference_mode)
        
        # 수치 안정성 검증
        if self.config.numerical_stability:
            predictions = self._ensure_numerical_stability(predictions)
        
        return predictions
    
    def _ensure_numerical_stability(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """수치 안정성 보장"""
        stable_predictions = {}
        
        for key, value in predictions.items():
            if torch.isnan(value).any() or torch.isinf(value).any():
                print(f"⚠️ {key}에서 NaN/Inf 감지, 0으로 대체")
                stable_predictions[key] = torch.zeros_like(value)
            else:
                stable_predictions[key] = value
        
        return stable_predictions
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """SOTA 손실 함수 사용"""
        # Pass expressibility mean and std from prediction_head to loss_function
        exp_mean = getattr(self.prediction_head, 'exp_mean', None)
        exp_std = getattr(self.prediction_head, 'exp_std', None)
        
        total_loss, individual_losses = self.loss_function(
            predictions, 
            targets, 
            exp_mean=exp_mean,
            exp_std=exp_std
        )
        
        return {
            'total': total_loss,
            **individual_losses
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'UnifiedPropertyPredictionTransformer',
            'architecture': 'SOTA Unified (Graph + Transformer + Cross-Attention)',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': {
                'd_model': self.config.d_model,
                'n_layers': self.config.n_layers,
                'n_heads': self.config.n_heads,
                'cross_attention_heads': self.config.cross_attention_heads,
                'consistency_loss_weight': self.config.consistency_loss_weight,
                'dropout': self.config.dropout
            },
            'features': [
                'Multi-Scale Feature Extraction',
                'Cross-Property Attention',
                'Property-Specific Decoders',
                'Consistency Loss',
                'Advanced Regularization',
                'Numerical Stability'
            ]
        }


# 레거시 호환성을 위한 별칭
IntegratedPropertyPredictionTransformer = UnifiedPropertyPredictionTransformer
IntegratedPropertyPredictionConfig = UnifiedPropertyPredictionConfig
PropertyPredictionTransformer = UnifiedPropertyPredictionTransformer
PropertyPredictionConfig = UnifiedPropertyPredictionConfig

# 손실 함수 별칭
IntegratedPropertyPredictionLoss = OptimalPropertyLoss
PropertyPredictionLoss = OptimalPropertyLoss


def create_property_prediction_model(config: Optional[UnifiedPropertyPredictionConfig] = None) -> UnifiedPropertyPredictionTransformer:
    """통합된 Property Prediction 모델 생성"""
    if config is None:
        config = UnifiedPropertyPredictionConfig()  # 통합 설정에서 기본값 사용
    
    model = UnifiedPropertyPredictionTransformer(config)
    
    print("🚀 통합된 Property Prediction Transformer 생성 완료!")
    print("=" * 60)
    
    model_info = model.get_model_info()
    print(f"📊 모델 정보:")
    print(f"   • 아키텍처: {model_info['architecture']}")
    print(f"   • 총 파라미터: {model_info['total_parameters']:,}")
    print(f"   • 학습 가능 파라미터: {model_info['trainable_parameters']:,}")
    print(f"   • d_model: {model_info['config']['d_model']}")
    print(f"   • n_layers: {model_info['config']['n_layers']}")
    print(f"   • n_heads: {model_info['config']['n_heads']}")
    
    print(f"\n🎯 SOTA 기능:")
    for feature in model_info['features']:
        print(f"   ✅ {feature}")
    
    return model


# 레거시 호환성 함수
create_integrated_model = create_property_prediction_model


if __name__ == "__main__":
    # 테스트 실행
    print("🧪 통합된 Property Prediction 모델 테스트")
    
    config = UnifiedPropertyPredictionConfig()  # 통합 설정에서 기본값 사용
    
    model = create_property_prediction_model(config)
    print("\n✅ 통합된 Property Prediction Transformer 테스트 완료!")
