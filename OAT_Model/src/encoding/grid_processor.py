"""
그리드 프로세서 - 그리드 매트릭스 구조 처리 전담 클래스
"""

import torch
from typing import Dict, List, Any, Tuple

class GridProcessor:
    """그리드 매트릭스 구조를 처리하고 상태 벡터로 변환"""
    
    def process_grid_matrix(self, grid_matrix: List[List], grid_shape: List[int]) -> Dict[str, Any]:
        """
        그리드 매트릭스를 상태 벡터들로 변환
        
        Args:
            grid_matrix: [num_qubits][max_parallel_order] 그리드 매트릭스
            grid_shape: [max_parallel_order, num_qubits]
            
        Returns:
            처리된 그리드 정보 (상태 벡터, 위치 정보 등)
        """
        max_parallel_order, num_qubits = grid_shape[0], grid_shape[1]
        
        if max_parallel_order == 0 or num_qubits == 0:
            return self._create_empty_grid_info()
        
        grid_states = []
        position_embeddings = []
        grid_info = {
            'gate_positions': [],  # 실제 게이트가 있는 위치들
            'grid_shape': grid_shape,
            'total_positions': max_parallel_order * num_qubits
        }
        
        # 2D 그리드를 순회하면서 상태 벡터 생성
        for qubit_idx in range(num_qubits):
            for time_step in range(max_parallel_order):
                node = grid_matrix[qubit_idx][time_step] if time_step < len(grid_matrix[qubit_idx]) else None
                
                if node and hasattr(node, 'gate_type'):
                    # 실제 게이트가 있는 경우
                    gate_type_id = getattr(node, 'gate_type_id', 0)
                    qubits = getattr(node, 'qubits', [qubit_idx])
                    parameter = getattr(node, 'parameter', 0.0)
                    
                    qubit1 = qubits[0] if len(qubits) > 0 else qubit_idx
                    qubit2 = qubits[1] if len(qubits) > 1 else qubit1
                    
                    # 컨트롤/타겟 역할 결정
                    is_control = float(qubit_idx == qubit1 and len(qubits) > 1)
                    is_target = float(qubit_idx == qubit2 and len(qubits) > 1)
                    
                    state_vector = [
                        float(gate_type_id), 
                        float(qubit1), 
                        float(qubit2), 
                        float(parameter or 0.0),
                        float(time_step),      # 시간 위치 정보
                        float(qubit_idx),      # 큐빗 위치 정보
                        is_control,            # 컨트롤 역할
                        is_target              # 타겟 역할
                    ]
                    
                    grid_info['gate_positions'].append((time_step, qubit_idx))
                else:
                    # 빈 위치는 패딩 토큰으로 처리
                    state_vector = [0.0, 0.0, 0.0, 0.0, float(time_step), float(qubit_idx), 0.0, 0.0]
                
                grid_states.append(state_vector)
                position_embeddings.append([float(time_step), float(qubit_idx)])
        
        return {
            'state_vectors': torch.tensor(grid_states, dtype=torch.float32),  # [grid_size, 8]
            'position_embeddings': torch.tensor(position_embeddings, dtype=torch.float32),  # [grid_size, 2]
            'grid_info': grid_info
        }
    
    def _create_empty_grid_info(self) -> Dict[str, Any]:
        """빈 그리드 정보 생성"""
        return {
            'state_vectors': torch.zeros(1, 8, dtype=torch.float32),
            'position_embeddings': torch.zeros(1, 2, dtype=torch.float32),
            'grid_info': {
                'gate_positions': [],
                'grid_shape': [0, 0],
                'total_positions': 0
            }
        }
