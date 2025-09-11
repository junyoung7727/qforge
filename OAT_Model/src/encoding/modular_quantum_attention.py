"""
Modular Quantum Attention System
기존 임베딩 파이프라인과 호환되는 모듈러 어텐션 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

class AttentionMode(Enum):
    """어텐션 모드 정의"""
    STANDARD = "standard"
    ADVANCED = "advanced"
    
    @classmethod
    def from_string(cls, mode_str: str):
        """문자열로부터 AttentionMode 생성"""
        mode_str = mode_str.lower()
        for mode in cls:
            if mode.value == mode_str:
                return mode
        raise ValueError(f"Unknown attention mode: {mode_str}. Available modes: {[m.value for m in cls]}")

class StandardQuantumAttention(nn.Module):
    """표준 양자 어텐션 - 기존 시스템과 호환"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # 표준 멀티헤드 어텐션
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 게이트 타입별 기본 가중치
        self.gate_weights = nn.Parameter(torch.ones(20))
        
        # 레이어 정규화
        self.norm = nn.LayerNorm(d_model)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """어텐션 모듈 가중치 초기화"""
        # MultiheadAttention의 가중치 초기화
        for name, param in self.self_attention.named_parameters():
            if 'weight' in name:
                if 'in_proj' in name or 'out_proj' in name:
                    # Query, Key, Value projection 및 output projection
                    torch.nn.init.xavier_uniform_(param, gain=1.0/math.sqrt(2))
                else:
                    torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.normal_(param, 0.0, 0.01)
        
        # 게이트 가중치 초기화
        torch.nn.init.normal_(self.gate_weights, 1.0, 0.1)
        
        # LayerNorm 초기화
        torch.nn.init.ones_(self.norm.weight)
        torch.nn.init.zeros_(self.norm.bias)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                gate_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len] or [batch_size, seq_len, seq_len] 
            gate_types: [batch_size, seq_len]
        """
        # 게이트 타입별 가중치 적용
        if gate_types is not None:
            gate_weights = self.gate_weights[gate_types.long().clamp(0, 19)]
            x = x * gate_weights.unsqueeze(-1)
        
        # attention mask 차원 변환
        # TorchScript 호환성: 항상 적절한 Tensor로 초기화
        batch_size, seq_len = x.shape[:2]
        
        if attention_mask is not None:
            if attention_mask.dim() == 2:  # [batch_size, seq_len]
                # PyTorch MultiHeadAttention은 key_padding_mask를 사용
                # key_padding_mask: [batch_size, seq_len] where True means ignore
                processed_mask = (attention_mask == 0)  # 패딩 위치는 True
            elif attention_mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                # 2D attention mask는 지원하지 않으므로 key_padding_mask로 변환
                processed_mask = (attention_mask.sum(dim=-1) == 0)
            else:
                # 예상치 못한 차원의 경우 기본값 사용
                processed_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
        else:
            # attention_mask가 None인 경우 모든 위치를 유효하게 처리
            processed_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
        
        # 표준 셀프 어텐션 (key_padding_mask 사용)
        # TorchScript 호환성을 위해 명시적으로 키워드 인자 사용
        attn_output, _ = self.self_attention(
            query=x, 
            key=x, 
            value=x, 
            key_padding_mask=processed_mask
        )
        
        # 잔차 연결 및 정규화
        return self.norm(x + attn_output)

class AdvancedQuantumAttention(nn.Module):
    """고급 양자 어텐션 - 얽힘, 공간적, 중요도 기반"""
    
    def __init__(self, d_model: int, num_heads: int = 8, max_qubits: int = 20):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_qubits = max_qubits
        
        # 얽힘 인식 어텐션
        self.entanglement_attention = self._create_entanglement_attention()
        
        # 공간적 어텐션
        self.spatial_attention = self._create_spatial_attention()
        
        # 중요도 기반 어텐션
        self.importance_attention = self._create_importance_attention()
        
        # 어텐션 융합
        self.attention_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 게이트별 어텐션 가중치
        self.gate_attention_weights = nn.Parameter(torch.ones(20, 3))
        
        # 레이어 정규화
        self.norm = nn.LayerNorm(d_model)
        
    def _create_entanglement_attention(self):
        """얽힘 인식 어텐션 생성"""
        return nn.Sequential(
            nn.MultiheadAttention(self.d_model, self.num_heads, dropout=0.1, batch_first=True),
        )
    
    def _create_spatial_attention(self):
        """공간적 어텐션 생성"""
        return nn.Sequential(
            nn.MultiheadAttention(self.d_model, self.num_heads, dropout=0.1, batch_first=True),
        )
    
    def _create_importance_attention(self):
        """중요도 기반 어텐션 생성"""
        return nn.Sequential(
            nn.MultiheadAttention(self.d_model, self.num_heads, dropout=0.1, batch_first=True),
        )
    
    def forward(self, x: torch.Tensor, graph_data: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            graph_data: 그래프 관련 정보 (선택적)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 얽힘 어텐션 - TorchScript 호환성을 위해 명시적 키워드 인자 사용
        entanglement_out, _ = self.entanglement_attention[0](
            query=x, key=x, value=x
        )
        
        # 공간적 어텐션 - TorchScript 호환성을 위해 명시적 키워드 인자 사용
        spatial_out, _ = self.spatial_attention[0](
            query=x, key=x, value=x
        )
        
        # 중요도 어텐션 - TorchScript 호환성을 위해 명시적 키워드 인자 사용
        importance_out, _ = self.importance_attention[0](
            query=x, key=x, value=x
        )
        
        # 게이트 타입별 가중치 적용
        if graph_data and 'gate_types' in graph_data:
            gate_types = graph_data['gate_types'].long().clamp(0, 19)
            weights = self.gate_attention_weights[gate_types]  # [batch, seq, 3]
            
            entanglement_out = entanglement_out * weights[:, :, 0:1]
            spatial_out = spatial_out * weights[:, :, 1:2]
            importance_out = importance_out * weights[:, :, 2:3]
        
        # 어텐션 융합 - pre-allocate to avoid fragmentation
        combined_dim = entanglement_out.size(-1) + spatial_out.size(-1) + importance_out.size(-1)
        combined = torch.empty(*entanglement_out.shape[:-1], combined_dim, 
                              device=entanglement_out.device, dtype=entanglement_out.dtype)
        combined[..., :entanglement_out.size(-1)] = entanglement_out
        combined[..., entanglement_out.size(-1):entanglement_out.size(-1)+spatial_out.size(-1)] = spatial_out
        combined[..., entanglement_out.size(-1)+spatial_out.size(-1):] = importance_out
        fused_output = self.attention_fusion(combined)
        
        # 잔차 연결 및 정규화
        return self.norm(x + fused_output)

class ModularQuantumAttention(nn.Module):
    """모듈러 양자 어텐션 - 모드별 전환 가능"""
    
    def __init__(self, d_model: int, num_heads: int = 8, max_qubits: int = 20, 
                 mode: AttentionMode = AttentionMode.STANDARD):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.mode = mode
        
        # 모드별 어텐션 모듈
        self.standard_attention = StandardQuantumAttention(d_model, num_heads)
        self.advanced_attention = AdvancedQuantumAttention(d_model, num_heads, max_qubits)
        
        # 크로스 어텐션 (Pre-Norm 구조)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Pre-Norm 레이어들
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # FFN (Pre-Norm 구조)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
        
    def set_mode(self, mode: AttentionMode):
        """어텐션 모드 변경"""
        self.mode = mode
        
    def forward(self, x: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                graph_data: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Pre-Norm 트랜스포머 구조
        Args:
            x: [batch_size, seq_len, d_model] - 쿼리
            context: [batch_size, ctx_len, d_model] - 크로스 어텐션용 컨텍스트 (선택적)
            attention_mask: 어텐션 마스크
            graph_data: 그래프 데이터 (고급 모드용)
        """
        # 1. Self-Attention (Pre-Norm)
        norm_x = self.norm1(x)
        
        if self.mode == AttentionMode.STANDARD:
            gate_types = graph_data['gate_types'] if graph_data else None
            attn_out = self.standard_attention(norm_x, attention_mask, gate_types)
        else:  # ADVANCED
            attn_out = self.advanced_attention(norm_x, graph_data)
        
        x = x + attn_out
        
        # 2. Cross-Attention (Pre-Norm) - 컨텍스트가 있는 경우
        if context is not None:
            norm_x = self.norm2(x)
            # TorchScript 호환성을 위해 명시적 키워드 인자 사용
            cross_out, _ = self.cross_attention(
                query=norm_x, 
                key=context, 
                value=context, 
                attn_mask=attention_mask
            )
            x = x + cross_out
        
        # 3. FFN (Pre-Norm)
        norm_x = self.norm3(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out
        
        return x

class QuantumTransformerLayer(nn.Module):
    """양자 트랜스포머 레이어 - 기존 Property Prediction 모델과 호환"""
    
    def __init__(self, d_model: int, num_heads: int = 8, max_qubits: int = 20,
                 attention_mode: AttentionMode = AttentionMode.STANDARD):
        super().__init__()
        self.d_model = d_model
        
        # 모듈러 어텐션
        self.attention = ModularQuantumAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_qubits=max_qubits,
            mode=attention_mode
        )
        
        # 드롭아웃
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                graph_data: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len, seq_len]
            context: [batch_size, ctx_len, d_model] - 크로스 어텐션용
            graph_data: 그래프 정보
        """
        output = self.attention(x, context, attention_mask, graph_data)
        return self.dropout(output)

class QuantumEmbeddingWithAttention(nn.Module):
    """어텐션이 통합된 양자 임베딩 시스템"""
    
    def __init__(self, d_model: int, config: Dict[str, Any]):
        super().__init__()
        self.d_model = d_model
        self.config = config
        
        # 어텐션 모드 설정
        attention_mode_str = config['attention_mode']
        self.attention_mode = AttentionMode.STANDARD if attention_mode_str == 'standard' else AttentionMode.ADVANCED
        
        # 기본 임베딩 레이어들
        self.node_embedding = nn.Sequential(
            nn.Linear(12, d_model // 2),  # 확장된 노드 특성
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 위치 임베딩
        self.position_embedding = nn.Embedding(1000, d_model)  # 최대 1000 시간 스텝
        self.qubit_embedding = nn.Embedding(config['max_qubits'], d_model)
        
        # 양자 트랜스포머 레이어들
        num_layers = config['num_attention_layers']
        self.transformer_layers = nn.ModuleList([
            QuantumTransformerLayer(
                d_model=d_model,
                num_heads=config['num_heads'],
                max_qubits=config['max_qubits'],
                attention_mode=self.attention_mode
            ) for _ in range(num_layers)
        ])
        
        # 출력 투영
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, node_features: torch.Tensor,
                grid_positions: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            node_features: [batch_size, seq_len, 12] - 확장된 노드 특성
            grid_positions: [batch_size, seq_len, 4] - [t, q, control_id, target_id]
            attention_mask: [batch_size, seq_len, seq_len]
            context: [batch_size, ctx_len, d_model] - 크로스 어텐션용
        """
        batch_size, seq_len, _ = node_features.shape
        
        # 1. 노드 특성 임베딩
        x = self.node_embedding(node_features)
        
        # 2. 위치 임베딩 추가
        if grid_positions is not None:
            time_pos = grid_positions[:, :, 0].long().clamp(0, 999)
            qubit_pos = grid_positions[:, :, 1].long().clamp(0, self.config['max_qubits'] - 1)
            
            pos_emb = self.position_embedding(time_pos)
            qubit_emb = self.qubit_embedding(qubit_pos)
            
            x = x + pos_emb + qubit_emb
        
        # 3. 그래프 데이터 준비
        graph_data = None
        if grid_positions is not None:
            gate_types = node_features[:, :, 0]  # 첫 번째 특성이 게이트 타입
            graph_data = {
                'gate_types': gate_types,
                'grid_positions': grid_positions,
                'node_features': node_features
            }
        
        # 4. 트랜스포머 레이어들 적용
        for layer in self.transformer_layers:
            x = layer(x, attention_mask, context, graph_data)
        
        # 5. 출력 투영
        output = self.output_projection(x)
        
        return output
    
    def set_attention_mode(self, mode: AttentionMode):
        """모든 레이어의 어텐션 모드 변경"""
        self.attention_mode = mode
        for layer in self.transformer_layers:
            layer.attention.set_mode(mode)

class PropertyPredictionCompatibleEncoder(nn.Module):
    """Property Prediction 모델과 호환되는 인코더"""
    
    def __init__(self, d_model: int, config: Dict[str, Any]):
        super().__init__()
        self.d_model = d_model
        
        # 어텐션 통합 임베딩
        self.embedding_with_attention = QuantumEmbeddingWithAttention(d_model, config)
        
        # 기존 Property Prediction 모델과의 호환성을 위한 출력 형식 조정
        self.compatibility_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
    def forward(self, circuit_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        기존 Property Prediction 모델과 호환되는 인터페이스
        Args:
            circuit_data: 회로 데이터 딕셔너리
        Returns:
            기존 모델이 기대하는 형식의 출력
        """
        # 노드 특성과 그리드 위치 추출
        node_features = circuit_data['node_features']
        grid_positions = circuit_data['grid_positions']
        attention_mask = circuit_data['attention_mask']
        
        # 어텐션 통합 임베딩 적용
        embeddings = self.embedding_with_attention(
            node_features=node_features,
            grid_positions=grid_positions,
            attention_mask=attention_mask
        )
        
        # 호환성 레이어 적용
        compatible_embeddings = self.compatibility_layer(embeddings)
        
        return {
            'embeddings': compatible_embeddings,
            'attention_mask': attention_mask,
            'sequence_length': embeddings.size(1)
        }
