"""
인코딩 파이프라인 팩토리
Decision Transformer와 Property Prediction을 위한 별도 파이프라인 생성
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union
from enum import Enum

from .decision_transformer_encoder import DecisionTransformerEncoder
from .property_prediction_encoder import PropertyPredictionEncoder
from .base_quantum_encoder import BaseQuantumEncoder, EncodingResult, QuantumCircuitGraphBuilder

class EncodingMode(Enum):
    """인코딩 모드 정의"""
    DECISION_TRANSFORMER = "decision_transformer"
    PROPERTY_PREDICTION = "property_prediction"
    HYBRID = "hybrid"

class EncodingPipelineFactory:
    """인코딩 파이프라인 팩토리 클래스"""
    
    def __init__(self):
        """팩토리 초기화"""
        self.graph_builder = QuantumCircuitGraphBuilder()
    
    @staticmethod
    def create_encoder(mode: EncodingMode, d_model: int, config: Dict[str, Any]) -> BaseQuantumEncoder:
        """지정된 모드에 따른 인코더 생성"""
        
        if mode == EncodingMode.DECISION_TRANSFORMER:
            return DecisionTransformerEncoder(d_model, config)
        
        elif mode == EncodingMode.PROPERTY_PREDICTION:
            return PropertyPredictionEncoder(d_model, config)
        
        elif mode == EncodingMode.HYBRID:
            return HybridQuantumEncoder(d_model, config)
        
        else:
            raise ValueError(f"Unsupported encoding mode: {mode}")
    
    @staticmethod
    def create_pipeline(mode: EncodingMode, d_model: int, config: Dict[str, Any]) -> 'EncodingPipeline':
        """완전한 인코딩 파이프라인 생성"""
        encoder = EncodingPipelineFactory.create_encoder(mode, d_model, config)
        return EncodingPipeline(encoder, mode, config)

class HybridQuantumEncoder(BaseQuantumEncoder):
    """Decision Transformer와 Property Prediction을 모두 지원하는 하이브리드 인코더"""
    
    def __init__(self, d_model: int, config: Dict[str, Any]):
        super().__init__(d_model, config)
        
        # 두 인코더 모두 생성
        self.dt_encoder = DecisionTransformerEncoder(d_model, config)
        self.pp_encoder = PropertyPredictionEncoder(d_model, config)
        
        # 공유 백본 네트워크
        self.shared_backbone = self.dt_encoder.gnn  # GNN 백본 공유
        
        # 모드별 어댑터
        self.dt_adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        self.pp_adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
    def encode_circuit(self, circuit_spec) -> EncodingResult:
        """기본 회로 인코딩 (Property Prediction 모드)"""
        return self.pp_encoder.encode_circuit(circuit_spec)
    
    def encode_for_decision_transformer(self, circuit_spec, current_step: int) -> EncodingResult:
        """Decision Transformer용 인코딩"""
        # 공유 백본으로 기본 인코딩
        base_result = self.dt_encoder.encode_for_action_prediction(circuit_spec, current_step)
        
        # DT 어댑터 적용
        adapted_embeddings = self.dt_adapter(base_result.embeddings)
        
        return EncodingResult(
            embeddings=adapted_embeddings,
            graph_data=base_result.graph_data,
            auxiliary_data={
                **base_result.auxiliary_data,
                'encoder_type': 'hybrid_dt'
            }
        )
    
    def encode_for_property_prediction(self, circuit_spec) -> EncodingResult:
        """Property Prediction용 인코딩"""
        # 공유 백본으로 기본 인코딩
        base_result = self.pp_encoder.encode_for_property_prediction(circuit_spec)
        
        # PP 어댑터 적용
        adapted_embeddings = self.pp_adapter(base_result.embeddings)
        
        return EncodingResult(
            embeddings=adapted_embeddings,
            graph_data=base_result.graph_data,
            auxiliary_data={
                **base_result.auxiliary_data,
                'encoder_type': 'hybrid_pp'
            }
        )

class EncodingPipeline:
    """통합 인코딩 파이프라인"""
    
    def __init__(self, encoder: BaseQuantumEncoder, mode: EncodingMode, config: Dict[str, Any]):
        self.encoder = encoder
        self.mode = mode
        self.config = config
        
        # 캐싱 설정
        self.enable_caching = config.get('enable_caching', True)
        self.cache = {} if self.enable_caching else None
        
        # 배치 처리 설정
        self.batch_size = config.get('batch_size', 32)
        
    def encode_single(self, circuit_spec, **kwargs) -> EncodingResult:
        """단일 회로 인코딩"""
        cache_key = None
        
        if self.enable_caching:
            cache_key = self._generate_cache_key(circuit_spec, kwargs)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # 모드에 따른 인코딩
        if self.mode == EncodingMode.DECISION_TRANSFORMER:
            current_step = kwargs.get('current_step', 0)
            result = self.encoder.encode_for_action_prediction(circuit_spec, current_step)
        
        elif self.mode == EncodingMode.PROPERTY_PREDICTION:
            result = self.encoder.encode_for_property_prediction(circuit_spec)
        
        elif self.mode == EncodingMode.HYBRID:
            task_type = kwargs.get('task_type', 'property_prediction')
            if task_type == 'decision_transformer':
                current_step = kwargs.get('current_step', 0)
                result = self.encoder.encode_for_decision_transformer(circuit_spec, current_step)
            else:
                result = self.encoder.encode_for_property_prediction(circuit_spec)
        
        else:
            result = self.encoder.encode_circuit(circuit_spec)
        
        # 캐싱 - 메모리 제한 적용
        if self.enable_caching and cache_key:
            # 캐시 크기 제한 (최대 100개 항목)
            if len(self.cache) >= 100:
                # 가장 오래된 항목 제거 (FIFO)
                oldest_key = next(iter(self.cache))
                old_result = self.cache.pop(oldest_key)
                # 메모리 해제
                if hasattr(old_result, 'embeddings') and hasattr(old_result.embeddings, 'data'):
                    del old_result.embeddings
            self.cache[cache_key] = result
        
        return result
    
    def encode_batch(self, circuit_specs: List, **kwargs) -> List[EncodingResult]:
        """배치 회로 인코딩"""
        results = []
        
        # 배치 크기로 분할 처리
        for i in range(0, len(circuit_specs), self.batch_size):
            batch = circuit_specs[i:i + self.batch_size]
            batch_results = []
            
            for spec in batch:
                result = self.encode_single(spec, **kwargs)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def encode_incremental_sequence(self, circuit_spec, max_steps: Optional[int] = None) -> List[EncodingResult]:
        """증분적 시퀀스 인코딩 (Decision Transformer용)"""
        if self.mode not in [EncodingMode.DECISION_TRANSFORMER, EncodingMode.HYBRID]:
            raise ValueError(f"Incremental encoding not supported for mode: {self.mode}")
        
        sequence_length = len(circuit_spec.gates)
        if max_steps is not None:
            sequence_length = min(sequence_length, max_steps)
        
        results = []
        for step in range(sequence_length + 1):
            if self.mode == EncodingMode.HYBRID:
                result = self.encode_single(circuit_spec, current_step=step, task_type='decision_transformer')
            else:
                result = self.encode_single(circuit_spec, current_step=step)
            results.append(result)
        
        return results
    
    def analyze_property_trends(self, circuit_specs: List) -> Dict[str, Any]:
        """속성 트렌드 분석 (Property Prediction용)"""
        if self.mode not in [EncodingMode.PROPERTY_PREDICTION, EncodingMode.HYBRID]:
            raise ValueError(f"Property analysis not supported for mode: {self.mode}")
        
        results = self.encode_batch(circuit_specs)
        
        # 속성 예측값 수집
        property_predictions = {}
        for result in results:
            if 'property_predictions' in result.auxiliary_data:
                preds = result.auxiliary_data['property_predictions']
                for prop_name, pred_value in preds.items():
                    if prop_name not in property_predictions:
                        property_predictions[prop_name] = []
                    property_predictions[prop_name].append(pred_value.item())
        
        # 통계 계산
        statistics = {}
        for prop_name, values in property_predictions.items():
            values_tensor = torch.tensor(values)
            statistics[prop_name] = {
                'mean': torch.mean(values_tensor).item(),
                'std': torch.std(values_tensor).item(),
                'min': torch.min(values_tensor).item(),
                'max': torch.max(values_tensor).item(),
                'count': len(values)
            }
        
        return {
            'statistics': statistics,
            'raw_predictions': property_predictions,
            'num_circuits': len(circuit_specs)
        }
    
    def _generate_cache_key(self, circuit_spec, kwargs: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        # 회로 ID와 주요 파라미터 기반 키 생성
        key_parts = [
            circuit_spec.circuit_id,
            str(self.mode.value),
            str(kwargs.get('current_step', '')),
            str(kwargs.get('task_type', ''))
        ]
        return '_'.join(filter(None, key_parts))
    
    def clear_cache(self):
        """캐시 클리어 - 메모리 누수 방지"""
        if self.cache:
            # 캐시된 텐서들의 메모리 해제
            for result in self.cache.values():
                if hasattr(result, 'embeddings') and hasattr(result.embeddings, 'data'):
                    del result.embeddings
                if hasattr(result, 'auxiliary_data'):
                    for key, value in result.auxiliary_data.items():
                        if hasattr(value, 'data'):
                            del value
            self.cache.clear()
            # CUDA 캐시 정리
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """캐시 통계 - 메모리 사용량 추적"""
        if not self.cache:
            return {'cache_enabled': False}
        
        total_memory = 0
        for result in self.cache.values():
            if hasattr(result, 'embeddings') and result.embeddings is not None:
                total_memory += result.embeddings.numel() * result.embeddings.element_size()
            if hasattr(result, 'auxiliary_data'):
                for value in result.auxiliary_data.values():
                    if hasattr(value, 'numel') and hasattr(value, 'element_size'):
                        total_memory += value.numel() * value.element_size()
        
        return {
            'cache_enabled': True,
            'cache_size': len(self.cache),
            'cache_memory_mb': total_memory / (1024 * 1024)
        }
