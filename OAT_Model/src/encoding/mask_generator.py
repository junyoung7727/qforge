"""
마스크 생성기 - 어텐션 마스크와 액션 마스크 생성 전담 클래스
"""

import torch
from typing import Dict, List, Any

class MaskGenerator:
    """그리드 구조 기반 마스크 생성"""
    
    def create_all_masks(self, grid_info: Dict[str, Any], original_gate_count: int) -> Dict[str, torch.Tensor]:
        """
        모든 마스크를 생성
        
        Args:
            grid_info: 그리드 정보 (gate_positions, grid_shape 등)
            original_gate_count: 원본 게이트 수
            
        Returns:
            생성된 마스크들
        """
        grid_shape = grid_info['grid_shape']
        gate_positions = grid_info['gate_positions']
        total_positions = grid_info['total_positions']
        
        return {
            'attention_mask': self._create_attention_mask(grid_shape, total_positions),
            'action_mask': self._create_action_mask(gate_positions, total_positions)
        }
    
    def _create_attention_mask(self, grid_shape: List[int], total_positions: int) -> torch.Tensor:
        """
        2D 그리드 구조를 반영한 어텐션 마스크 생성
        
        Args:
            grid_shape: [max_parallel_order, num_qubits]
            total_positions: 전체 위치 수
            
        Returns:
            [total_positions, total_positions] 어텐션 마스크
        """
        if total_positions == 0:
            return torch.ones(1, 1, dtype=torch.bool)
        
        max_parallel_order, num_qubits = grid_shape
        attention_mask = torch.ones(total_positions, total_positions, dtype=torch.bool)
        
        # 2D 그리드 구조에서 인접한 위치들만 어텐션 허용
        for i in range(total_positions):
            for j in range(total_positions):
                # 위치를 2D 좌표로 변환
                i_time, i_qubit = divmod(i, num_qubits)
                j_time, j_qubit = divmod(j, num_qubits)
                
                # 시간적으로 이전 위치들과 같은 큐빗 라인의 인접 위치들에 어텐션 허용
                if (j_time <= i_time and 
                    (i_qubit == j_qubit or abs(i_time - j_time) <= 1)):
                    attention_mask[i, j] = True
                else:
                    attention_mask[i, j] = False
        
        return attention_mask
    
    def _create_action_mask(self, gate_positions: List[tuple], total_positions: int) -> torch.Tensor:
        """
        실제 게이트가 있는 위치만 액션 예측하도록 마스크 생성
        
        Args:
            gate_positions: 실제 게이트가 있는 (time_step, qubit_idx) 위치들
            total_positions: 전체 위치 수
            
        Returns:
            [total_positions] 액션 마스크
        """
        if total_positions == 0:
            return torch.zeros(1, dtype=torch.bool)
        
        action_mask = torch.zeros(total_positions, dtype=torch.bool)
        
        # 실제 게이트가 있는 위치만 True로 설정
        for time_step, qubit_idx in gate_positions:
            # 2D 좌표를 1D 인덱스로 변환
            position_idx = time_step * len(set(pos[1] for pos in gate_positions)) + qubit_idx
            if position_idx < total_positions:
                action_mask[position_idx] = True
        
        return action_mask
