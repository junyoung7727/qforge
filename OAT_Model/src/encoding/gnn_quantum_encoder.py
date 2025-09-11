"""
GNN-based Quantum Circuit Encoder
Graph Neural Network를 사용한 고수준 양자회로 임베딩
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphSAGE, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Any, Optional, Tuple
import math

from .base_quantum_encoder import BaseQuantumEncoder, QuantumGraphData, EncodingResult, QuantumCircuitGraphBuilder
from .modular_quantum_attention import ModularQuantumAttention, AttentionMode

class QuantumGNNLayer(nn.Module):
    """양자회로 특화 GNN 레이어"""
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, gnn_type: str = 'gat'):
        super().__init__()
        self.gnn_type = gnn_type
        self.out_dim = out_dim
        
        if gnn_type == 'gat':
            self.conv = GATConv(in_dim, out_dim // num_heads, heads=num_heads, dropout=0.1)
        elif gnn_type == 'sage':
            self.conv = GraphSAGE(in_dim, out_dim, num_layers=2, dropout=0.1)
        elif gnn_type == 'gcn':
            self.conv = GCNConv(in_dim, out_dim)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.1)
        
        # 모듈러 양자 어텐션 메커니즘
        self.quantum_attention = ModularQuantumAttention(
            d_model=out_dim, 
            num_heads=num_heads,
            mode=AttentionMode.STANDARD  # 기본값은 표준 모드
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # GNN 연산
        if self.gnn_type == 'sage':
            h = self.conv(x, edge_index)
        else:
            h = self.conv(x, edge_index, edge_attr)
        
        # 정규화 및 드롭아웃
        h = self.norm(h)
        h = self.dropout(h)
        
        # 모듈러 양자 어텐션 적용
        # 노드 특성에서 그래프 데이터 구성
        if x.size(-1) >= 12:
            batch_size = 1
            seq_len = h.size(0)
            
            # 그래프 데이터 구성
            graph_data = {
                'gate_types': x[:, 0].unsqueeze(0),  # [1, seq_len]
                'grid_positions': x[:, :4].unsqueeze(0),  # [1, seq_len, 4]
                'node_features': x.unsqueeze(0),  # [1, seq_len, 12]
                'edge_index': edge_index
            }
            
            h_attended = self.quantum_attention(
                h.unsqueeze(0),  # [1, seq_len, d_model]
                graph_data=graph_data
            ).squeeze(0)
        else:
            # 기본 어텐션만 적용
            h_attended = self.quantum_attention(
                h.unsqueeze(0)
            ).squeeze(0)
            
        h = h + h_attended  # 잔차 연결
        
        return h

# DEPRECATED: QuantumAwareAttention is replaced by ModularQuantumAttention
# This class is kept for backward compatibility only

class QuantumAwareAttention(nn.Module):
    """DEPRECATED: Use ModularQuantumAttention instead"""
    
    def __init__(self, d_model: int, max_qubits: int = 20):
        super().__init__()
        import warnings
        warnings.warn(
            "QuantumAwareAttention is deprecated. Use ModularQuantumAttention instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.d_model = d_model
        self.max_qubits = max_qubits
        
        # 모듈러 어텐션으로 대체
        self.modular_attention = ModularQuantumAttention(
            d_model=d_model, 
            num_heads=8, 
            max_qubits=max_qubits,
            mode=AttentionMode.STANDARD
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None, 
                node_features: Optional[torch.Tensor] = None,
                grid_positions: Optional[torch.Tensor] = None,
                gate_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # 그래프 데이터 준비
        if node_features is None:
            node_features = torch.zeros(x.size(0), x.size(1), 12, device=x.device)
        if grid_positions is None:
            grid_positions = torch.zeros(x.size(0), x.size(1), 4, device=x.device)
        if gate_types is None:
            gate_types = torch.zeros(x.size(0), x.size(1), device=x.device)
            
        graph_data = {
            'edge_index': edge_index,
            'node_features': node_features,
            'grid_positions': grid_positions,
            'gate_types': gate_types
        }
        
        # 통합 어텐션 적용
        attended_output = self.unified_attention(x, graph_data)
        
        # 잔차 연결
        residual_output = self.residual_projection(x)
        
        return attended_output + residual_output

class HierarchicalQuantumGNN(nn.Module):
    """계층적 양자회로 GNN"""
    
    def __init__(self, d_model: int, num_layers: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # 다중 스케일 GNN 레이어
        self.local_layers = nn.ModuleList([
            QuantumGNNLayer(d_model, d_model, gnn_type='gat') for _ in range(num_layers)
        ])
        
        self.global_layers = nn.ModuleList([
            QuantumGNNLayer(d_model, d_model, gnn_type='sage') for _ in range(num_layers)
        ])
        
        # 계층 간 융합
        self.fusion_layers = nn.ModuleList([
            nn.Linear(d_model * 2, d_model) for _ in range(num_layers)
        ])
        
        # 최종 출력 레이어
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x
        
        for i in range(self.num_layers):
            # 로컬 패턴 학습 (GAT)
            h_local = self.local_layers[i](h, edge_index, edge_attr)
            
            # 글로벌 패턴 학습 (GraphSAGE)
            h_global = self.global_layers[i](h, edge_index, edge_attr)
            
            # 로컬과 글로벌 정보 융합 - pre-allocate to avoid fragmentation
            combined_dim = h_local.size(-1) + h_global.size(-1)
            h_combined = torch.empty(h_local.size(0), combined_dim, device=h_local.device, dtype=h_local.dtype)
            h_combined[:, :h_local.size(-1)] = h_local
            h_combined[:, h_local.size(-1):] = h_global
            h = self.fusion_layers[i](h_combined)
            h = F.relu(h)
        
        # 최종 출력
        h = self.output_projection(h)
        
        return h

class GNNQuantumEncoder(BaseQuantumEncoder):
    """GNN 기반 양자회로 인코더"""
    
    def __init__(self, d_model: int, config: Dict[str, Any]):
        super().__init__(d_model, config)
        
        self.graph_builder = QuantumCircuitGraphBuilder()
        
        # 확장된 노드 특성 임베딩 (12차원)
        self.node_embedding = nn.Sequential(
            nn.Linear(12, d_model // 2),  # 12차원 확장 상태 벡터
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 확장된 엣지 특성 임베딩 (4차원)
        self.edge_embedding = nn.Sequential(
            nn.Linear(4, d_model // 4),  # [edge_type, weight, direction_info, gate_type]
            nn.LayerNorm(d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 2)
        )
        
        # 계층적 GNN
        self.gnn = HierarchicalQuantumGNN(d_model, num_layers=config.get('gnn_layers', 3))
        
        # 위치 인코딩 (그리드 구조 보존)
        self.positional_encoding = QuantumPositionalEncoding(d_model)
        
        # 풀링 레이어
        self.global_pool = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # mean + max pooling
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
    def encode_circuit(self, circuit_spec) -> EncodingResult:
        """전체 회로 인코딩"""
        # 그리드 매트릭스 생성
        from ..encoding.grid_graph_encoder import GridGraphEncoder
        grid_encoder = GridGraphEncoder()
        encoded_data = grid_encoder.encode(circuit_spec)
        grid_matrix_data = grid_encoder.to_grid_matrix(encoded_data)
        
        # 그래프 구조 생성
        graph_data = self.graph_builder.build_graph_from_grid(grid_matrix_data)
        
        # GNN 인코딩
        embeddings = self._encode_graph(graph_data)
        
        return EncodingResult(
            embeddings=embeddings,
            graph_data=graph_data,
            auxiliary_data={'encoding_type': 'full_circuit'}
        )
    
    def encode_partial_circuit(self, circuit_spec, mask: Optional[torch.Tensor] = None) -> EncodingResult:
        """부분 회로 인코딩 (Decision Transformer용)"""
        # 마스크 적용된 회로 생성
        if mask is not None:
            # 마스크에 따라 게이트 필터링
            filtered_gates = []
            for i, gate in enumerate(circuit_spec.gates):
                if i < len(mask) and mask[i]:
                    filtered_gates.append(gate)
            
            # 새로운 CircuitSpec 생성
            from ..data.quantum_circuit_dataset import CircuitSpec
            partial_spec = CircuitSpec(
                circuit_id=f"{circuit_spec.circuit_id}_partial",
                gates=filtered_gates,
                num_qubits=circuit_spec.num_qubits,
                properties=circuit_spec.properties
            )
        else:
            partial_spec = circuit_spec
        
        # 부분 회로 인코딩
        result = self.encode_circuit(partial_spec)
        result.auxiliary_data['encoding_type'] = 'partial_circuit'
        result.auxiliary_data['mask'] = mask
        
        return result
    
    def get_graph_representation(self, circuit_spec) -> QuantumGraphData:
        """그래프 표현 생성"""
        from ..encoding.grid_graph_encoder import GridGraphEncoder
        grid_encoder = GridGraphEncoder()
        encoded_data = grid_encoder.encode(circuit_spec)
        grid_matrix_data = grid_encoder.to_grid_matrix(encoded_data)
        
        return self.graph_builder.build_graph_from_grid(grid_matrix_data)
    
    def _encode_graph(self, graph_data: QuantumGraphData) -> torch.Tensor:
        """그래프 데이터를 임베딩으로 변환"""
        # 노드 특성 임베딩
        node_emb = self.node_embedding(graph_data.node_features)
        
        # 위치 인코딩 추가
        pos_emb = self.positional_encoding(graph_data.temporal_order, graph_data.grid_shape)
        node_emb = node_emb + pos_emb
        
        # 엣지 특성 임베딩
        edge_emb = None
        if graph_data.edge_features.size(0) > 0:
            edge_emb = self.edge_embedding(graph_data.edge_features)
        
        # GNN 적용
        graph_emb = self.gnn(node_emb, graph_data.edge_index, edge_emb)
        
        # 글로벌 풀링 - pre-allocate to avoid fragmentation
        mean_pool = torch.mean(graph_emb, dim=0, keepdim=True)
        max_pool = torch.max(graph_emb, dim=0, keepdim=True)[0]
        # Pre-allocate concatenated tensor
        pool_combined = torch.empty(mean_pool.size(0), mean_pool.size(-1) + max_pool.size(-1), 
                                   device=mean_pool.device, dtype=mean_pool.dtype)
        pool_combined[:, :mean_pool.size(-1)] = mean_pool
        pool_combined[:, mean_pool.size(-1):] = max_pool
        global_emb = self.global_pool(pool_combined)
        
        return global_emb

class QuantumPositionalEncoding(nn.Module):
    """양자회로 특화 위치 인코딩"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # 시간적 위치 인코딩
        self.temporal_encoding = nn.Embedding(1000, d_model // 2)
        
        # 큐비트 위치 인코딩
        self.qubit_encoding = nn.Embedding(100, d_model // 2)
        
    def forward(self, temporal_order: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        time_steps, num_qubits = grid_shape
        
        # 시간적 위치
        temporal_pos = temporal_order.long().clamp(0, 999)
        temp_emb = self.temporal_encoding(temporal_pos)
        
        # 큐비트 위치
        qubit_pos = torch.arange(len(temporal_order), device=temporal_order.device) % num_qubits
        qubit_emb = self.qubit_encoding(qubit_pos.clamp(0, 99))
        
        # 결합 - pre-allocate to avoid fragmentation
        combined_dim = temp_emb.size(-1) + qubit_emb.size(-1)
        pos_emb = torch.empty(*temp_emb.shape[:-1], combined_dim, device=temp_emb.device, dtype=temp_emb.dtype)
        pos_emb[..., :temp_emb.size(-1)] = temp_emb
        pos_emb[..., temp_emb.size(-1):] = qubit_emb
        
        return pos_emb
