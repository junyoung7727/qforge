"""
Property Prediction Model Integration
기존 Property Prediction 모델과 새로운 모듈러 어텐션 시스템 통합
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from .modular_quantum_attention import (
    AttentionMode, 
    ModularQuantumAttention, 
    QuantumEmbeddingWithAttention,
    PropertyPredictionCompatibleEncoder
)

class EnhancedPropertyPredictionEncoder(nn.Module):
    """향상된 Property Prediction 인코더 - 모듈러 어텐션 통합"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.d_model = config['d_model']
        self.max_qubits = config['max_qubits']
        
        # 어텐션 모드 설정
        attention_mode_str = config['attention_mode']
        self.attention_mode = AttentionMode.STANDARD if attention_mode_str == 'standard' else AttentionMode.ADVANCED
        
        # 호환성 인코더
        self.compatible_encoder = PropertyPredictionCompatibleEncoder(self.d_model, config)
        
        # Property-specific 헤드들
        self.property_heads = nn.ModuleDict({
            'entanglement': nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Linear(self.d_model // 2, 1),
                nn.Sigmoid()
            ),
            'expressibility': nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Linear(self.d_model // 2, 1),
                nn.Sigmoid()
            ),
            'fidelity': nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Linear(self.d_model // 2, 1),
                nn.Sigmoid()
            )
        })
        
        # 통합 예측 헤드
        self.unified_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 3)  # [entanglement, expressibility, fidelity]
        )
        
    def forward(self, circuit_data: Dict[str, torch.Tensor], 
                mode: str = 'unified') -> Dict[str, torch.Tensor]:
        """
        Args:
            circuit_data: 회로 데이터
            mode: 'unified', 'individual', 'both'
        """
        # 호환성 인코더로 임베딩 생성
        encoding_result = self.compatible_encoder(circuit_data)
        embeddings = encoding_result['embeddings']  # [batch, seq, d_model]
        
        # 글로벌 풀링으로 회로 레벨 표현 생성
        circuit_embedding = torch.mean(embeddings, dim=1)  # [batch, d_model]
        
        results = {}
        
        if mode in ['individual', 'both']:
            # 개별 property 예측
            for prop_name, head in self.property_heads.items():
                results[prop_name] = head(circuit_embedding)
        
        if mode in ['unified', 'both']:
            # 통합 예측
            #print(f"DEBUG: Creating unified predictions with mode={mode}")
            unified_output = self.unified_head(circuit_embedding)
            #print(f"DEBUG: unified_output shape: {unified_output.shape}")
            results['unified'] = {
                'entanglement': unified_output[:, 0:1],
                'expressibility': unified_output[:, 1:2], 
                'fidelity': unified_output[:, 2:3]
            }
            #print(f"DEBUG: Added unified key to results")
        
        # 메타데이터 추가
        results['embeddings'] = embeddings
        results['circuit_embedding'] = circuit_embedding
        results['attention_mode'] = self.attention_mode.value
        
        return results
    
    def set_attention_mode(self, mode_str: str):
        """어텐션 모드 변경 (문자열 입력 지원)"""
        if isinstance(mode_str, str):
            mode = AttentionMode.STANDARD if mode_str == 'standard' else AttentionMode.ADVANCED
        else:
            mode = mode_str
        
        self.attention_mode = mode
        self.compatible_encoder.embedding_with_attention.set_attention_mode(mode)
        print(f"EnhancedPropertyPredictionEncoder attention mode changed to: {mode.value}")

class LegacyCompatibilityAdapter(nn.Module):
    """기존 모델과의 호환성을 위한 어댑터"""
    
    def __init__(self, legacy_model: nn.Module, enhanced_encoder: EnhancedPropertyPredictionEncoder):
        super().__init__()
        self.legacy_model = legacy_model
        self.enhanced_encoder = enhanced_encoder
        
        # 출력 형식 변환기
        self.output_converter = nn.Linear(
            enhanced_encoder.d_model, 
            self._get_legacy_output_dim()
        )
        
    def _get_legacy_output_dim(self) -> int:
        """기존 모델의 출력 차원 추정"""
        # 기존 모델의 마지막 레이어 차원을 찾아서 반환
        for module in reversed(list(self.legacy_model.modules())):
            if isinstance(module, nn.Linear):
                return module.out_features
        return 256  # 기본값
    
    def forward(self, circuit_data: Dict[str, torch.Tensor], 
                use_enhanced: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            circuit_data: 회로 데이터
            use_enhanced: True면 향상된 인코더 사용, False면 기존 모델 사용
        """
        if use_enhanced:
            # 향상된 인코더 사용
            enhanced_result = self.enhanced_encoder(circuit_data, mode='both')
            
            # 기존 모델 형식으로 변환
            circuit_embedding = enhanced_result['circuit_embedding']
            legacy_format = self.output_converter(circuit_embedding)
            
            return {
                'legacy_output': legacy_format,
                'enhanced_output': enhanced_result,
                'mode': 'enhanced'
            }
        else:
            # 기존 모델 사용 (호환성 테스트용)
            legacy_output = self.legacy_model(circuit_data)
            enhanced_result = self.enhanced_encoder(circuit_data, mode='unified')
            return {
                'legacy_output': legacy_output,
                'enhanced_output': enhanced_result,
                'mode': 'legacy'
            }

def create_integrated_property_predictor(config: Dict[str, Any], 
                                       legacy_model: Optional[nn.Module] = None) -> nn.Module:
    """통합 Property Prediction 모델 생성"""
    
    # 향상된 인코더 생성
    enhanced_encoder = EnhancedPropertyPredictionEncoder(config)
    
    # 어텐션 모드 설정 확인 및 적용
    attention_mode = config['attention_mode']
    enhanced_encoder.set_attention_mode(attention_mode)
    
    print(f"Created integrated property predictor with attention mode: {attention_mode}")
    
    if legacy_model is not None:
        # 기존 모델이 있으면 호환성 어댑터만 생성 (마이그레이션 매니저 제거)
        adapter = LegacyCompatibilityAdapter(legacy_model, enhanced_encoder)
        return adapter
    else:
        # 새로운 모델만 사용 (권장)
        return enhanced_encoder

# 사용 예시를 위한 설정
DEFAULT_CONFIG = {
    'd_model': 256,
    'num_heads': 8,
    'max_qubits': 20,
    'num_attention_layers': 3,
    'attention_mode': 'standard'  # 'standard' or 'advanced'
}
