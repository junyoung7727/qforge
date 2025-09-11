"""
공통 양자 회로 데이터 구조 정의
순환 임포트를 방지하기 위한 기본 데이터 클래스 모듈
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import re
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from gates import GateOperation


def extract_depth_from_circuit_id(circuit_id: str) -> int:
    """
    circuit_id에서 depth 값을 추출합니다.
    
    예시: "scalability_test1_16q_d1_r0.3_0" -> 1
          "circuit_8q_d10_r0.5_2" -> 10
    
    Args:
        circuit_id: 회로 ID 문자열
        
    Returns:
        depth 값 (찾을 수 없으면 0)
    """
    # _d뒤에 오는 숫자를 찾는 정규식
    pattern = r'_d(\d+)_'
    match = re.search(pattern, circuit_id)
    
    if match:
        return int(match.group(1))
    else:
        # 패턴을 찾을 수 없으면 0 반환
        print(f"Warning: Could not extract depth from circuit_id: {circuit_id}")
        return 0


@dataclass
class CircuitSpec:
    """양자 회로 스펙"""
    circuit_id: str
    num_qubits: int
    gates: List[GateOperation]
    depth: int = 0  # 기본값 추가
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CircuitSpec':
        """딕셔너리에서 CircuitSpec 생성"""
        gates = [
            GateOperation(
                name=gate['name'],
                qubits=gate['qubits'],
                parameters=gate['parameters']
            )
            for gate in data['gates']
        ]
        
        # circuit_id에서 depth 추출
        circuit_id = data['circuit_id']
        depth = extract_depth_from_circuit_id(circuit_id)
        
        return cls(
            circuit_id=circuit_id,
            num_qubits=data['num_qubits'],
            gates=gates,
            depth=depth
        )


@dataclass
class MeasurementResult:
    """측정 결과 데이터 - timestamp 제외하고 모든 필드 포함"""
    circuit_id: str
    num_qubits: int
    depth: int
    fidelity: float
    robust_fidelity: Optional[float] = None
    expressibility: Optional[Dict[str, float]] = None
    entanglement: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MeasurementResult':
        """딕셔너리에서 MeasurementResult 생성
        
        예:
        {
            "circuit_id": "scalability_test_15q_d1_r0.8_0",
            "num_qubits": 15,
            "depth": 16,
            "timestamp": "2025-08-13T10:14:44.745735",
            "fidelity": 0.73046875,
            "robust_fidelity": 0.962890625,
            "expressibility": {
                "kl_divergence": 8.466604598985924
            },
            "entanglement": 0.5020833333333334
        }
        """
        return cls(
            circuit_id=data['circuit_id'],
            num_qubits=data['num_qubits'],
            depth=data['depth'],
            fidelity=data['fidelity'],
            robust_fidelity=data.get('robust_fidelity'),
            expressibility=data.get('expressibility'),
            entanglement=data.get('entanglement')
        )


@dataclass
class CircuitData:
    """회로 스펙과 측정 결과를 결합한 데이터"""
    circuit_spec: CircuitSpec
    measurement_result: MeasurementResult
    
    @property
    def circuit_id(self) -> str:
        return self.circuit_spec.circuit_id
    
    @property
    def num_qubits(self) -> int:
        return self.circuit_spec.num_qubits
    
    @property
    def gates(self) -> List[GateOperation]:
        return self.circuit_spec.gates


class PropertyNormalizer:
    """양자 회로 속성 정규화 유틸리티"""
    
    def __init__(self):
        self.entanglement_median = None
        self.entanglement_q25 = None
        self.entanglement_q75 = None
        self.entanglement_iqr = None
        self.is_fit = False
    
    def fit(self, data: List[Dict[str, Any]]) -> None:
        """scalability 데이터에서 얽힘도 통계값 계산 (Robust 정규화)"""
        import numpy as np
        
        entanglement_values = []
        
        for item in data:
            if 'entanglement' in item and item['entanglement'] is not None:
                entanglement_values.append(float(item['entanglement']))
        
        if entanglement_values:
            entanglement_array = np.array(entanglement_values)
            self.entanglement_median = float(np.median(entanglement_array))
            self.entanglement_q25 = float(np.percentile(entanglement_array, 25))
            self.entanglement_q75 = float(np.percentile(entanglement_array, 75))
            self.entanglement_iqr = self.entanglement_q75 - self.entanglement_q25
            
            # IQR이 0이면 표준편차 기반으로 대체
            if self.entanglement_iqr == 0:
                std = float(np.std(entanglement_array))
                self.entanglement_iqr = 2 * std if std > 0 else 1.0
        else:
            self.entanglement_median = 0.5
            self.entanglement_q25 = 0.0
            self.entanglement_q75 = 1.0
            self.entanglement_iqr = 1.0
        
        self.is_fit = True
        print(f"Entanglement robust stats - Median: {self.entanglement_median:.4f}, IQR: {self.entanglement_iqr:.4f}")
        print(f"Q25: {self.entanglement_q25:.4f}, Q75: {self.entanglement_q75:.4f}")
    
    def normalize_entanglement(self, value: Optional[float]) -> Optional[float]:
        """얽힘도 값을 Robust 정규화 + Sigmoid 스케일링"""
        if not self.is_fit:
            # 정규화 전에 fit이 호출되지 않았다면 클리핑으로 대체
            return float(min(max(value or 0.0, 0.0), 1.0)) if value is not None else None
        
        if value is None:
            return None
        
        import numpy as np
        
        value = float(value)
        
        # IQR이 0이거나 매우 작으면 단순 클리핑 사용
        if self.entanglement_iqr <= 1e-10:
            return float(min(max(value, 0.0), 1.0))
        
        # Robust 정규화: (x - median) / IQR
        robust_normalized = (value - self.entanglement_median) / self.entanglement_iqr
        
        # 극값 클리핑으로 infinity 방지
        robust_normalized = np.clip(robust_normalized, -10.0, 10.0)
        
        # Sigmoid 함수로 0-1 범위로 부드럽게 매핑
        sigmoid_value = 1 / (1 + np.exp(-robust_normalized))
        
        return float(sigmoid_value)
