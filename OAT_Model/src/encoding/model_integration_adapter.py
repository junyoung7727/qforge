"""
기존 모델과의 통합을 위한 어댑터
새로운 GNN 기반 인코더를 기존 Property Prediction Transformer와 통합
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import warnings

from .encoding_pipeline_factory import EncodingPipelineFactory, EncodingMode
from .base_quantum_encoder import EncodingResult

class LegacyEmbeddingAdapter:
    """기존 임베딩 파이프라인과의 호환성을 위한 어댑터"""
    
    def __init__(self, new_encoder, legacy_facade=None):
        self.new_encoder = new_encoder
        self.legacy_facade = legacy_facade
        self.compatibility_mode = legacy_facade is not None
        
    def process_circuit_spec(self, circuit_spec, use_new_encoder: bool = True) -> Dict[str, Any]:
        """CircuitSpec을 처리하여 임베딩 생성"""
        
        if use_new_encoder:
            # 새로운 GNN 인코더 사용
            result = self.new_encoder.encode_for_property_prediction(circuit_spec)
            
            return {
                'embeddings': result.embeddings,
                'graph_data': result.graph_data,
                'property_predictions': result.auxiliary_data['property_predictions'],
                'encoding_type': 'gnn_based'
            }
        else:
            # 기존 레거시 파이프라인 사용
            if not self.compatibility_mode:
                raise ValueError("Legacy facade not available")
            
            # 기존 그리드 기반 인코딩
            from ..encoding.grid_graph_encoder import GridGraphEncoder
            grid_encoder = GridGraphEncoder()
            encoded_data = grid_encoder.encode(circuit_spec)
            grid_matrix_data = grid_encoder.to_grid_matrix(encoded_data)
            
            # 레거시 facade 호출
            legacy_result = self.legacy_facade.process_grid_matrix_data(
                grid_matrix_data.grid_matrix,
                grid_matrix_data.gate_sequence,
                grid_matrix_data.metadata
            )
            
            return {
                'embeddings': legacy_result['state_embeddings'],
                'masks': legacy_result['masks'],
                'encoding_type': 'legacy_grid'
            }

class PropertyPredictionModelAdapter(nn.Module):
    """Property Prediction Transformer와의 통합 어댑터"""
    
    def __init__(self, d_model: int, config: Dict[str, Any]):
        super().__init__()
        self.d_model = d_model
        self.config = config
        
        # 새로운 GNN 인코더 생성
        self.gnn_encoder = EncodingPipelineFactory.create_encoder(
            EncodingMode.PROPERTY_PREDICTION, 
            d_model, 
            config
        )
        
        # 기존 모델과의 호환성을 위한 차원 매핑
        self.dimension_adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 기존 임베딩 형식으로 변환
        self.format_converter = EmbeddingFormatConverter(d_model)
        
        # 하이브리드 모드 설정
        self.hybrid_mode = config['hybrid_mode']
        self.legacy_weight = config['legacy_weight']
        
    def forward(self, circuit_spec, use_gnn: bool = True) -> Dict[str, torch.Tensor]:
        """순전파 - 기존 모델 인터페이스 유지"""
        
        if use_gnn:
            # 새로운 GNN 기반 인코딩
            result = self.gnn_encoder.encode_for_property_prediction(circuit_spec)
            
            # 차원 어댑터 적용
            adapted_embeddings = self.dimension_adapter(result.embeddings)
            
            # 기존 형식으로 변환
            formatted_output = self.format_converter.convert_to_legacy_format(
                adapted_embeddings, result.graph_data
            )
            
            # 속성 예측 결과 추가
            if 'property_predictions' in result.auxiliary_data:
                formatted_output.update(result.auxiliary_data['property_predictions'])
            
            return formatted_output
        
        else:
            # 기존 방식으로 폴백
            warnings.warn("Falling back to legacy encoding method")
            return self._legacy_forward(circuit_spec)
    
    def _legacy_forward(self, circuit_spec) -> Dict[str, torch.Tensor]:
        """기존 방식의 순전파 (폴백용)"""
        # 기존 그리드 인코딩 방식 사용
        from ..encoding.grid_graph_encoder import GridGraphEncoder
        
        grid_encoder = GridGraphEncoder()
        encoded_data = grid_encoder.encode(circuit_spec)
        grid_matrix_data = grid_encoder.to_grid_matrix(encoded_data)
        
        # 간단한 임베딩 생성 (실제로는 기존 facade 사용해야 함)
        # Remove dummy embeddings - this should not be used
        raise NotImplementedError("Legacy fallback should not be used")
        
        return {
            'embeddings': dummy_embeddings,
            'encoding_type': 'legacy_fallback'
        }

class EmbeddingFormatConverter:
    """임베딩 형식 변환기"""
    
    def __init__(self, d_model: int):
        self.d_model = d_model
        
    def convert_to_legacy_format(self, gnn_embeddings: torch.Tensor, graph_data) -> Dict[str, torch.Tensor]:
        """GNN 임베딩을 기존 형식으로 변환"""
        
        # 기존 Property Prediction Transformer가 기대하는 형식
        batch_size = gnn_embeddings.size(0)
        
        # 시퀀스 길이 추정 (그래프 노드 수 기반)
        seq_len = graph_data.node_features.size(0) if graph_data else 1
        
        # 시퀀스 형태로 재구성
        if gnn_embeddings.dim() == 2:
            # [batch_size, d_model] -> [batch_size, seq_len, d_model]
            sequence_embeddings = gnn_embeddings.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            sequence_embeddings = gnn_embeddings
        
        # 어텐션 마스크 생성
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=gnn_embeddings.device)
        
        return {
            'embeddings': sequence_embeddings,
            'attention_mask': attention_mask,
            'sequence_length': seq_len
        }
    
    def convert_from_legacy_format(self, legacy_embeddings: torch.Tensor) -> torch.Tensor:
        """기존 형식을 GNN 형식으로 변환"""
        
        if legacy_embeddings.dim() == 3:
            # [batch_size, seq_len, d_model] -> [batch_size, d_model]
            return torch.mean(legacy_embeddings, dim=1)
        
        return legacy_embeddings

class HybridTrainingManager:
    """하이브리드 학습 관리자"""
    
    def __init__(self, gnn_encoder, legacy_encoder=None, config: Dict[str, Any] = None):
        self.gnn_encoder = gnn_encoder
        self.legacy_encoder = legacy_encoder
        self.config = config or {}
        
        # 학습 스케줄링 설정
        self.warmup_epochs = self.config['warmup_epochs']
        self.transition_epochs = self.config['transition_epochs']
        self.gnn_weight_schedule = self._create_weight_schedule()
        
    def _create_weight_schedule(self) -> List[float]:
        """GNN 가중치 스케줄 생성"""
        schedule = []
        
        # 워밍업 단계: 기존 방식 위주
        for epoch in range(self.warmup_epochs):
            weight = 0.1 + 0.4 * (epoch / self.warmup_epochs)
            schedule.append(weight)
        
        # 전환 단계: 점진적으로 GNN 비중 증가
        for epoch in range(self.transition_epochs):
            weight = 0.5 + 0.5 * (epoch / self.transition_epochs)
            schedule.append(weight)
        
        # 이후: GNN 위주
        return schedule
    
    def get_encoding_weights(self, epoch: int) -> Tuple[float, float]:
        """현재 에포크의 인코딩 가중치 반환"""
        if epoch < len(self.gnn_weight_schedule):
            gnn_weight = self.gnn_weight_schedule[epoch]
        else:
            gnn_weight = 1.0
        
        legacy_weight = 1.0 - gnn_weight
        return gnn_weight, legacy_weight
    
    def hybrid_encode(self, circuit_spec, epoch: int) -> Dict[str, torch.Tensor]:
        """하이브리드 인코딩 수행"""
        gnn_weight, legacy_weight = self.get_encoding_weights(epoch)
        
        # GNN 인코딩
        gnn_result = self.gnn_encoder.encode_for_property_prediction(circuit_spec)
        gnn_embeddings = gnn_result.embeddings
        
        if self.legacy_encoder and legacy_weight > 0:
            # 레거시 인코딩
            legacy_result = self.legacy_encoder.encode_circuit(circuit_spec)
            legacy_embeddings = legacy_result.embeddings
            
            # 차원 맞추기
            if gnn_embeddings.shape != legacy_embeddings.shape:
                converter = EmbeddingFormatConverter(gnn_embeddings.size(-1))
                legacy_embeddings = converter.convert_from_legacy_format(legacy_embeddings)
            
            # 가중 평균
            combined_embeddings = (
                gnn_weight * gnn_embeddings + 
                legacy_weight * legacy_embeddings
            )
        else:
            combined_embeddings = gnn_embeddings
        
        return {
            'embeddings': combined_embeddings,
            'gnn_weight': gnn_weight,
            'legacy_weight': legacy_weight,
            'property_predictions': gnn_result.auxiliary_data['property_predictions']
        }

class BackwardCompatibilityWrapper:
    """기존 코드와의 하위 호환성 래퍼"""
    
    def __init__(self, new_pipeline, legacy_facade=None):
        self.new_pipeline = new_pipeline
        self.legacy_facade = legacy_facade
        
    def process_single_circuit(self, circuit_spec, **kwargs):
        """기존 embedding_pipeline.py의 process_single_circuit 인터페이스 유지"""
        
        # 새로운 파이프라인으로 처리
        result = self.new_pipeline.encode_single(circuit_spec, **kwargs)
        
        # 기존 형식으로 변환
        converter = EmbeddingFormatConverter(result.embeddings.size(-1))
        legacy_format = converter.convert_to_legacy_format(
            result.embeddings, 
            result.graph_data
        )
        
        return {
            'state_embeddings': legacy_format['embeddings'],
            'attention_mask': legacy_format['attention_mask'],
            'graph_data': result.graph_data,
            'encoding_metadata': {
                'encoding_type': result.auxiliary_data['encoding_type'],
                'property_predictions': result.auxiliary_data['property_predictions']
            }
        }
    
    def process_batch(self, circuit_specs: List, **kwargs):
        """배치 처리 인터페이스"""
        results = []
        
        for spec in circuit_specs:
            result = self.process_single_circuit(spec, **kwargs)
            results.append(result)
        
        return results
