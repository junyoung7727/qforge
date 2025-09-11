"""
Property Prediction 전용 인코더
양자회로 속성 예측에 특화된 GNN 기반 인코더
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import math

from .gnn_quantum_encoder import GNNQuantumEncoder
from .base_quantum_encoder import EncodingResult, QuantumGraphData

class PropertyPredictionHead(nn.Module):
    """양자회로 속성 예측을 위한 전용 헤드"""
    
    def __init__(self, d_model: int, property_config: Dict[str, Any]):
        super().__init__()
        self.d_model = d_model
        self.property_config = property_config
        
        # 공통 특성 추출기
        self.shared_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU()
        )
        
        # 속성별 예측 헤드
        self.property_heads = nn.ModuleDict()
        
        # Entanglement 예측 헤드
        if 'entanglement' in property_config:
            self.property_heads['entanglement'] = nn.Sequential(
                nn.Linear(d_model // 2, d_model // 4),
                nn.LayerNorm(d_model // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 4, 1)
                # 시그모이드는 손실 함수에서 적용
            )
        
        # Expressibility 예측 헤드
        if 'expressibility' in property_config:
            self.property_heads['expressibility'] = nn.Sequential(
                nn.Linear(d_model // 2, d_model // 4),
                nn.LayerNorm(d_model // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 4, 1)
            )
        
        # Fidelity 예측 헤드
        if 'fidelity' in property_config:
            self.property_heads['fidelity'] = nn.Sequential(
                nn.Linear(d_model // 2, d_model // 4),
                nn.LayerNorm(d_model // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid()  # [0, 1] 범위
            )
        
        # 추가 속성들을 위한 일반적 헤드
        for prop_name, prop_info in property_config.items():
            if prop_name not in ['entanglement', 'expressibility', 'fidelity']:
                activation = None
                if prop_info['range'] == [0, 1]:
                    activation = nn.Sigmoid()
                elif prop_info['range'] and len(prop_info['range']) == 2:
                    # 범위가 지정된 경우 tanh + scaling
                    activation = nn.Tanh()
                
                layers = [
                    nn.Linear(d_model // 2, d_model // 4),
                    nn.LayerNorm(d_model // 4),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model // 4, 1)
                ]
                if activation:
                    layers.append(activation)
                
                self.property_heads[prop_name] = nn.Sequential(*layers)
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """속성 예측 수행"""
        # 공통 특성 추출
        shared_features = self.shared_encoder(embeddings)
        
        # 각 속성별 예측
        predictions = {}
        for prop_name, head in self.property_heads.items():
            predictions[prop_name] = head(shared_features)
        
        return predictions

class QuantumStateAnalyzer(nn.Module):
    """양자 상태 분석을 위한 전용 모듈"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # 큐비트 간 상관관계 분석
        self.correlation_analyzer = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 얽힘 패턴 감지
        self.entanglement_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU()
        )
        
        # 회로 복잡도 분석
        self.complexity_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU()
        )
        
    def forward(self, graph_embeddings: torch.Tensor, graph_data: QuantumGraphData) -> Dict[str, torch.Tensor]:
        """양자 상태 분석 수행"""
        batch_size = graph_embeddings.size(0)
        
        # 큐비트 간 상관관계 분석 - TorchScript 호환성을 위해 명시적 키워드 인자 사용
        corr_output, corr_weights = self.correlation_analyzer(
            query=graph_embeddings, 
            key=graph_embeddings, 
            value=graph_embeddings
        )
        
        # 얽힘 패턴 감지
        entanglement_features = self.entanglement_detector(corr_output)
        
        # 회로 복잡도 분석
        complexity_features = self.complexity_analyzer(graph_embeddings)
        
        return {
            'correlation_features': corr_output,
            'correlation_weights': corr_weights,
            'entanglement_features': entanglement_features,
            'complexity_features': complexity_features
        }

class PropertyPredictionEncoder(GNNQuantumEncoder):
    """Property Prediction 전용 GNN 인코더"""
    
    def __init__(self, d_model: int, config: Dict[str, Any]):
        super().__init__(d_model, config)
        
        # 속성 예측 설정
        self.property_config = config['properties']
        
        # 속성 예측 헤드
        self.property_head = PropertyPredictionHead(d_model, self.property_config)
        
        # 양자 상태 분석기
        self.state_analyzer = QuantumStateAnalyzer(d_model)
        
        # 속성 간 상관관계 모델링
        self.property_correlation = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 앙상블 예측을 위한 다중 인코더
        self.ensemble_encoders = nn.ModuleList([
            self._create_ensemble_encoder() for _ in range(3)
        ])
        
        # 앙상블 결합 레이어
        self.ensemble_combiner = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def _create_ensemble_encoder(self) -> nn.Module:
        """앙상블을 위한 개별 인코더 생성"""
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Dropout(0.15),  # 다양성을 위해 높은 드롭아웃
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU()
        )
    
    def encode_for_property_prediction(self, circuit_spec) -> EncodingResult:
        """속성 예측을 위한 인코딩"""
        # 전체 회로 그래프 생성
        graph_data = self.get_graph_representation(circuit_spec)
        
        # GNN으로 기본 그래프 인코딩
        base_embedding = self._encode_graph(graph_data)
        
        # 앙상블 인코딩
        ensemble_embeddings = []
        for encoder in self.ensemble_encoders:
            ensemble_emb = encoder(base_embedding)
            ensemble_embeddings.append(ensemble_emb)
        
        # 앙상블 결합
        combined_embedding = torch.cat(ensemble_embeddings, dim=-1)
        final_embedding = self.ensemble_combiner(combined_embedding)
        
        # 양자 상태 분석
        state_analysis = self.state_analyzer(final_embedding.unsqueeze(0), graph_data)
        
        # 속성 예측
        property_predictions = self.property_head(final_embedding)
        
        return EncodingResult(
            embeddings=final_embedding,
            graph_data=graph_data,
            auxiliary_data={
                'property_predictions': property_predictions,
                'state_analysis': state_analysis,
                'ensemble_embeddings': ensemble_embeddings,
                'encoding_type': 'property_prediction'
            }
        )
    
    def encode_with_uncertainty(self, circuit_spec, n_samples: int = 10) -> EncodingResult:
        """불확실성 추정을 포함한 인코딩"""
        # 드롭아웃을 활성화하여 여러 번 예측
        self.train()  # 드롭아웃 활성화
        
        predictions = []
        embeddings = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                result = self.encode_for_property_prediction(circuit_spec)
                predictions.append(result.auxiliary_data['property_predictions'])
                embeddings.append(result.embeddings)
        
        # 예측 통계 계산
        pred_stats = {}
        for prop_name in predictions[0].keys():
            prop_preds = torch.stack([p[prop_name] for p in predictions])
            pred_stats[prop_name] = {
                'mean': torch.mean(prop_preds, dim=0),
                'std': torch.std(prop_preds, dim=0),
                'min': torch.min(prop_preds, dim=0)[0],
                'max': torch.max(prop_preds, dim=0)[0]
            }
        
        # 임베딩 통계
        emb_stack = torch.stack(embeddings)
        mean_embedding = torch.mean(emb_stack, dim=0)
        
        self.eval()  # 다시 평가 모드로
        
        return EncodingResult(
            embeddings=mean_embedding,
            auxiliary_data={
                'uncertainty_predictions': pred_stats,
                'n_samples': n_samples,
                'encoding_type': 'property_prediction_uncertainty'
            }
        )
    
    def analyze_property_correlations(self, circuit_specs: List) -> Dict[str, torch.Tensor]:
        """여러 회로의 속성 간 상관관계 분석"""
        all_predictions = []
        all_embeddings = []
        
        for spec in circuit_specs:
            result = self.encode_for_property_prediction(spec)
            all_predictions.append(result.auxiliary_data['property_predictions'])
            all_embeddings.append(result.embeddings)
        
        # 예측값들을 스택
        stacked_predictions = {}
        for prop_name in all_predictions[0].keys():
            stacked_predictions[prop_name] = torch.stack([
                p[prop_name] for p in all_predictions
            ])
        
        # 상관관계 계산
        correlations = {}
        prop_names = list(stacked_predictions.keys())
        
        for i, prop1 in enumerate(prop_names):
            for j, prop2 in enumerate(prop_names[i+1:], i+1):
                pred1 = stacked_predictions[prop1].flatten()
                pred2 = stacked_predictions[prop2].flatten()
                
                # 피어슨 상관계수 계산
                corr = torch.corrcoef(torch.stack([pred1, pred2]))[0, 1]
                correlations[f"{prop1}_{prop2}"] = corr
        
        return {
            'correlations': correlations,
            'predictions': stacked_predictions,
            'embeddings': torch.stack(all_embeddings)
        }
