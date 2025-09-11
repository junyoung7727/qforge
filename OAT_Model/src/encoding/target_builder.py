"""
타겟 빌더 - 학습용 타겟 생성 전담 클래스
"""

import torch
from typing import Dict, List, Any

class TargetBuilder:
    """학습용 타겟 (액션, 위치, 파라미터) 생성"""
    
    def create_targets(self, grid_info: Dict[str, Any], original_gate_count: int) -> Dict[str, torch.Tensor]:
        """
        학습용 타겟들을 생성
        
        Args:
            grid_info: 그리드 정보 (gate_positions, grid_shape 등)
            original_gate_count: 원본 게이트 수
            
        Returns:
            생성된 타겟들 (gate_targets, position_targets, parameter_targets)
        """
        gate_positions = grid_info['gate_positions']
        total_positions = grid_info['total_positions']
        
        if total_positions == 0:
            return self._create_empty_targets()
        
        # 타겟 텐서 초기화
        gate_targets = torch.zeros(total_positions, dtype=torch.long)
        position_targets = torch.full((total_positions, 2), -1, dtype=torch.long)
        parameter_targets = torch.zeros(total_positions, dtype=torch.float)
        
        # 실제 게이트가 있는 위치에만 타겟 설정
        for i, (time_step, qubit_idx) in enumerate(gate_positions):
            if i < original_gate_count:  # 원본 게이트 수만큼만
                # 위치 인덱스 계산
                position_idx = time_step * grid_info['grid_shape'][1] + qubit_idx
                if position_idx < total_positions:
                    # 여기서는 기본값으로 설정 (실제로는 그리드 매트릭스에서 추출해야 함)
                    gate_targets[position_idx] = 1  # 기본 게이트 타입
                    position_targets[position_idx] = torch.tensor([qubit_idx, qubit_idx], dtype=torch.long)
                    parameter_targets[position_idx] = 0.0  # 기본 파라미터
        
        return {
            'gate_targets': gate_targets,
            'position_targets': position_targets,
            'parameter_targets': parameter_targets
        }
    
    def _create_empty_targets(self) -> Dict[str, torch.Tensor]:
        """빈 타겟 생성"""
        return {
            'gate_targets': torch.zeros(1, dtype=torch.long),
            'position_targets': torch.full((1, 2), -1, dtype=torch.long),
            'parameter_targets': torch.zeros(1, dtype=torch.float)
        }
