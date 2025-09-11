"""
Base Quantum Circuit Encoder Interface
공통 인터페이스 정의 - Decision Transformer와 Property Prediction 인코더의 기본 클래스
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class QuantumGraphData:
    """양자회로 그래프 데이터 구조"""
    node_features: torch.Tensor      # [num_nodes, node_dim] - 노드 특성
    edge_index: torch.Tensor         # [2, num_edges] - 엣지 인덱스
    edge_features: torch.Tensor      # [num_edges, edge_dim] - 엣지 특성
    temporal_order: torch.Tensor     # [num_nodes] - 시간적 순서
    grid_shape: Tuple[int, int]      # 원본 그리드 형태 (time_steps, num_qubits)
    grid_positions: torch.Tensor     # [num_nodes, 2] - 그리드 위치 (time, qubit)
    node_types: torch.Tensor         # [num_nodes] - 노드 타입 (gate, qubit, etc.)
    attention_mask: Optional[torch.Tensor] = None  # [num_nodes] - 어텐션 마스크
    node_to_grid_mapping: Optional[Dict[int, Tuple[int, int]]] = None  # 노드 -> 그리드 위치
    grid_to_node_mapping: Optional[Dict[Tuple[int, int], int]] = None  # 그리드 위치 -> 노드
    
@dataclass
class EncodingResult:
    """인코딩 결과 구조"""
    embeddings: torch.Tensor         # 주요 임베딩 결과
    attention_mask: Optional[torch.Tensor] = None
    auxiliary_data: Optional[Dict[str, Any]] = None
    graph_data: Optional[QuantumGraphData] = None

class BaseQuantumEncoder(ABC, nn.Module):
    """양자회로 인코더 기본 클래스"""
    
    def __init__(self, d_model: int, config: Dict[str, Any]):
        super().__init__()
        self.d_model = d_model
        self.config = config
        
    @abstractmethod
    def encode_circuit(self, circuit_spec) -> EncodingResult:
        """회로 전체 인코딩"""
        pass
    
    @abstractmethod
    def encode_partial_circuit(self, circuit_spec, mask: Optional[torch.Tensor] = None) -> EncodingResult:
        """부분 회로 인코딩 (Decision Transformer용)"""
        pass
    
    @abstractmethod
    def get_graph_representation(self, circuit_spec) -> QuantumGraphData:
        """그래프 표현 생성"""
        pass

class QuantumCircuitGraphBuilder:
    """양자회로를 그래프 구조로 변환하는 빌더"""
    
    def __init__(self):
        self.node_types = {
            'gate': 0,
            'qubit': 1, 
            'measurement': 2,
            'barrier': 3
        }
        self.edge_types = {
            'temporal': 0,      # 시간적 순서
            'qubit_connection': 1,  # 큐비트 연결
            'control_target': 2,    # 제어-타겟 관계
            'entanglement': 3       # 얽힘 관계
        }
    
    def build_graph_from_grid(self, grid_matrix_data: Dict[str, Any]) -> QuantumGraphData:
        """그리드 매트릭스에서 그래프 구조 생성"""
        grid_matrix = grid_matrix_data['grid_matrix']
        grid_shape = grid_matrix_data.get('grid_shape', grid_matrix.shape)
        
        # 노드 생성: 각 그리드 위치를 노드로 변환
        nodes, node_features, node_types = self._create_nodes_from_grid(grid_matrix, grid_shape)
        
        # 엣지 생성: 시간적, 공간적 관계 모델링
        edge_index, edge_features = self._create_edges_from_grid(grid_matrix, grid_shape, nodes)
        
        # 매핑 정보 생성
        qubit_mapping, gate_mapping = self._create_mappings(nodes, grid_shape)
        
        # 시간적 순서 정보
        temporal_order = self._extract_temporal_order(grid_matrix, grid_shape)
        
        # 그리드 위치 정보 생성
        grid_positions = self._create_grid_positions(nodes, grid_shape)
        
        # 기본 attention_mask 생성 (모든 노드에 대해 1)
        num_nodes = node_features.shape[0] if node_features.numel() > 0 else 0
        attention_mask = torch.ones(num_nodes, dtype=torch.float)
        
        return QuantumGraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            temporal_order=temporal_order,
            grid_shape=grid_shape,
            grid_positions=grid_positions,
            node_types=node_types,
            attention_mask=attention_mask,
            node_to_grid_mapping=qubit_mapping,
            grid_to_node_mapping=gate_mapping
        )
    
    def _create_nodes_from_grid(self, grid_matrix: torch.Tensor, grid_shape: Tuple[int, int]) -> Tuple[List, torch.Tensor, torch.Tensor]:
        """그리드에서 노드 생성 - 확장된 특성으로 제어/타겟 구별 및 단일 큐비트 정보 보존"""
        time_steps, num_qubits = grid_shape
        nodes = []
        node_features_list = []
        node_types_list = []
        
        for t in range(time_steps):
            for q in range(num_qubits):
                if t < grid_matrix.shape[0] and q < grid_matrix.shape[1]:
                    # 그리드 위치의 게이트 정보 추출
                    gate_info = grid_matrix[t, q]
                    
                    # 노드 정보 생성
                    node = {
                        'position': (t, q),
                        'gate_type': gate_info[0].item() if len(gate_info) > 0 else 0,
                        'time_step': t,
                        'qubit_idx': q
                    }
                    nodes.append(node)
                    
                    # 확장된 노드 특성 벡터 생성 (12차원)
                    features = torch.zeros(12)
                    
                    # 기본 게이트 정보 (0-7)
                    if len(gate_info) >= 8:
                        features[:8] = gate_info[:8].clone()
                    else:
                        features[:len(gate_info)] = gate_info
                    
                    gate_type = features[0].item()
                    
                    # 제어/타겟 역할 분석 및 단일 큐비트 정보 보존
                    is_control, is_target, control_id, target_id = self._analyze_gate_roles(
                        gate_type, features, t, q, num_qubits
                    )
                    
                    # 확장 특성 추가 (8-11)
                    features[8] = is_control      # 제어 큐비트 여부
                    features[9] = is_target       # 타겟 큐비트 여부
                    features[10] = control_id     # 제어 큐비트 ID (-1 if N/A)
                    features[11] = target_id      # 타겟 큐비트 ID (-1 if N/A)
                    
                    node_features_list.append(features)
                    
                    # 노드 타입 결정 (더 세분화)
                    if gate_type == 0:  # 빈 위치
                        node_types_list.append(self.node_types['qubit'])
                    else:  # 게이트 존재
                        node_types_list.append(self.node_types['gate'])
        
        node_features = torch.stack(node_features_list) if node_features_list else torch.empty(0, 12)
        node_types = torch.tensor(node_types_list, dtype=torch.long)
        
        return nodes, node_features, node_types
    
    def _analyze_gate_roles(self, gate_type: float, features: torch.Tensor, t: int, q: int, num_qubits: int) -> Tuple[float, float, float, float]:
        """게이트 역할 분석 - 제어/타겟 구별 및 단일 큐비트 보존"""
        is_control = 0.0
        is_target = 0.0
        control_id = -1.0
        target_id = -1.0
        
        gate_type_int = int(gate_type)
        
        # 단일 큐비트 게이트 (1-7: I, X, Y, Z, H, RX, RY, RZ 등)
        if 1 <= gate_type_int <= 7:
            # 단일 큐비트는 자기 자신이 타겟
            is_target = 1.0
            target_id = float(q)
            
        # 2큐비트 제어 게이트 (8-12: CX, CY, CZ, CCX 등)
        elif 8 <= gate_type_int <= 12:
            qubit1 = int(features[1].item()) if len(features) > 1 else q
            qubit2 = int(features[2].item()) if len(features) > 2 else q
            
            # 현재 노드의 큐비트가 어떤 역할인지 판단
            if q == qubit1:  # 제어 큐비트
                is_control = 1.0
                control_id = float(qubit1)
                target_id = float(qubit2)
            elif q == qubit2:  # 타겟 큐비트
                is_target = 1.0
                control_id = float(qubit1)
                target_id = float(qubit2)
                
        # SWAP 게이트 (13): 양방향 대칭
        elif gate_type_int == 13:
            qubit1 = int(features[1].item()) if len(features) > 1 else q
            qubit2 = int(features[2].item()) if len(features) > 2 else q
            
            if q in [qubit1, qubit2]:
                is_control = 0.5  # SWAP은 대칭적
                is_target = 0.5
                control_id = float(qubit1)
                target_id = float(qubit2)
        
        return is_control, is_target, control_id, target_id
    
    def _create_edges_from_grid(self, grid_matrix: torch.Tensor, grid_shape: Tuple[int, int], nodes: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """그리드에서 엣지 생성"""
        time_steps, num_qubits = grid_shape
        edge_list = []
        edge_features_list = []
        
        # 시간적 엣지 (같은 큐비트의 연속된 시간 스텝)
        for q in range(num_qubits):
            for t in range(time_steps - 1):
                src_idx = t * num_qubits + q
                dst_idx = (t + 1) * num_qubits + q
                
                if src_idx < len(nodes) and dst_idx < len(nodes):
                    edge_list.append([src_idx, dst_idx])
                    edge_features_list.append([self.edge_types['temporal'], 1.0])  # [type, weight]
        
        # 공간적 엣지 (같은 시간의 인접 큐비트)
        for t in range(time_steps):
            for q in range(num_qubits - 1):
                src_idx = t * num_qubits + q
                dst_idx = t * num_qubits + (q + 1)
                
                if src_idx < len(nodes) and dst_idx < len(nodes):
                    edge_list.append([src_idx, dst_idx])
                    edge_list.append([dst_idx, src_idx])  # 양방향
                    edge_features_list.append([self.edge_types['qubit_connection'], 0.5])
                    edge_features_list.append([self.edge_types['qubit_connection'], 0.5])
        
        # 향상된 제어-타겟 엣지 (양방향 및 역할 구별)
        for t in range(time_steps):
            for q in range(num_qubits):
                node_idx = t * num_qubits + q
                if node_idx < len(nodes) and t < grid_matrix.shape[0] and q < grid_matrix.shape[1]:
                    gate_info = grid_matrix[t, q]
                    gate_type = int(gate_info[0].item()) if len(gate_info) > 0 else 0
                    
                    # 2큐비트 게이트 처리
                    if len(gate_info) >= 3 and 8 <= gate_type <= 13:
                        qubit1, qubit2 = int(gate_info[1].item()), int(gate_info[2].item())
                        if qubit1 != qubit2 and 0 <= qubit1 < num_qubits and 0 <= qubit2 < num_qubits:
                            control_idx = t * num_qubits + qubit1
                            target_idx = t * num_qubits + qubit2
                            
                            if control_idx < len(nodes) and target_idx < len(nodes):
                                # 제어 → 타겟 엣지
                                edge_list.append([control_idx, target_idx])
                                edge_features_list.append([
                                    self.edge_types['control_target'], 
                                    1.0,  # weight
                                    1.0,  # is_control_to_target
                                    float(gate_type)  # gate_type_info
                                ])
                                
                                # 타겟 ← 제어 역방향 엣지 (정보 흐름)
                                edge_list.append([target_idx, control_idx])
                                edge_features_list.append([
                                    self.edge_types['control_target'],
                                    0.8,  # 역방향은 약간 낮은 가중치
                                    0.0,  # is_target_to_control
                                    float(gate_type)
                                ])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_features = torch.tensor(edge_features_list, dtype=torch.float)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_features = torch.empty(0, 4, dtype=torch.float)  # 4차원으로 확장
        
        return edge_index, edge_features
    
    def _create_mappings(self, nodes: List, grid_shape: Tuple[int, int]) -> Tuple[Dict[int, int], Dict[int, int]]:
        """큐비트 및 게이트 매핑 생성"""
        time_steps, num_qubits = grid_shape
        qubit_mapping = {}
        gate_mapping = {}
        
        for idx, node in enumerate(nodes):
            qubit_idx = node['qubit_idx']
            if qubit_idx not in qubit_mapping:
                qubit_mapping[qubit_idx] = []
            qubit_mapping[qubit_idx].append(idx)
            
            if node['gate_type'] != 0:  # 게이트가 존재하는 경우
                gate_id = node['time_step'] * num_qubits + qubit_idx
                gate_mapping[gate_id] = idx
        
        return qubit_mapping, gate_mapping
    
    def _extract_temporal_order(self, grid_matrix: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
        """시간적 순서 정보 추출"""
        time_steps, num_qubits = grid_shape
        temporal_order = torch.zeros(time_steps * num_qubits)
        
        for t in range(time_steps):
            for q in range(num_qubits):
                idx = t * num_qubits + q
                if idx < len(temporal_order):
                    temporal_order[idx] = t
        
        return temporal_order
    
    def build_graph_from_circuit_spec(self, circuit_spec) -> 'QuantumGraphData':
        """CircuitSpec에서 그래프 데이터 생성"""
        if hasattr(circuit_spec, 'grid_matrix_data'):
            # 그리드 매트릭스 데이터가 있는 경우
            return self.build_graph_from_grid(circuit_spec.grid_matrix_data)
        elif hasattr(circuit_spec, 'gates') and hasattr(circuit_spec, 'num_qubits'):
            # 게이트 리스트에서 그리드 매트릭스 생성 후 그래프 빌드
            grid_matrix_data = self._create_grid_from_gates(circuit_spec.gates, circuit_spec.num_qubits)
            return self.build_graph_from_grid(grid_matrix_data)
        else:
            raise ValueError(f"Unsupported circuit_spec format: {type(circuit_spec)}")
    
    def _create_grid_from_gates(self, gates, num_qubits) -> Dict[str, Any]:
        """게이트 리스트에서 그리드 매트릭스 데이터 생성"""
        # 간단한 그리드 생성 (실제 구현은 더 복잡할 수 있음)
        max_time_steps = len(gates) if gates else 1
        grid_matrix = torch.zeros((max_time_steps, num_qubits, 8))  # 8차원 게이트 정보
        
        for t, gate in enumerate(gates):
            if hasattr(gate, 'qubits') and gate.qubits:
                for qubit_idx in gate.qubits:
                    if qubit_idx < num_qubits:
                        # 게이트 타입과 기본 정보 설정
                        gate_type = getattr(gate, 'gate_type', 1)
                        grid_matrix[t, qubit_idx, 0] = gate_type
                        grid_matrix[t, qubit_idx, 1] = 1.0  # 게이트 존재 플래그
        
        return {
            'grid_matrix': grid_matrix,
            'grid_shape': (max_time_steps, num_qubits)
        }
    
    def _create_grid_positions(self, nodes: List, grid_shape: Tuple[int, int]) -> torch.Tensor:
        """노드들의 그리드 위치 정보를 텐서로 생성"""
        num_nodes = len(nodes)
        grid_positions = torch.zeros((num_nodes, 2), dtype=torch.float)
        
        for idx, node in enumerate(nodes):
            time_step = node['time_step']
            qubit_idx = node['qubit_idx']
            grid_positions[idx, 0] = float(time_step)
            grid_positions[idx, 1] = float(qubit_idx)
        
        return grid_positions
