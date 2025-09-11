"""
Abstract Base Classes for Quantum Dataset Architecture
효율적이고 확장 가능한 퀀텀 데이터셋 추상화
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from torch.utils.data import Dataset
import torch
from dataclasses import dataclass
# 공통 데이터 구조를 새로운 모듈에서 가져옴
from .quantum_data_structures import CircuitData, CircuitSpec, MeasurementResult


class BaseQuantumDataset(Dataset, ABC):
    """추상 퀀텀 데이터셋 베이스 클래스"""
    
    def __init__(self, circuit_data: List[CircuitData]):
        self.circuit_data = circuit_data
    
    def __len__(self) -> int:
        return len(self.circuit_data)
    
    def __getitem__(self, idx: int) -> Any:
        """서브클래스에서 구현할 데이터 반환 형식"""
        return self._get_formatted_item(idx)
    
    @abstractmethod
    def _get_formatted_item(self, idx: int) -> Any:
        """각 데이터셋 타입별 포맷팅 로직"""
        pass
    
    @abstractmethod
    def get_target_format(self) -> str:
        """데이터셋 용도 반환: 'property', 'decision_transformer', 'gate_prediction'"""
        pass
    
    @abstractmethod
    def get_collate_fn(self) -> Callable:
        """용도에 맞는 collate function 반환"""
        pass
    
    def get_by_circuit_id(self, circuit_id: str) -> Optional[CircuitData]:
        """circuit_id로 데이터 검색 (공통 기능)"""
        for data in self.circuit_data:
            if data.circuit_id == circuit_id:
                return data
        return None
    
    def get_base_circuit_data(self, idx: int) -> CircuitData:
        """기본 회로 데이터 반환 (공통 기능)"""
        return self.circuit_data[idx]


class PropertyPredictionDataset(BaseQuantumDataset):
    """프로퍼티 예측용 데이터셋"""
    
    def _get_formatted_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """프로퍼티 예측용 포맷으로 데이터 반환"""
        circuit_data = self.circuit_data[idx]
        
        # 게이트 시퀀스를 인덱스로 변환
        from gates import gate_registry
        gate_indices = []
        qubit_positions = []
        gate_parameters = []
        
        for gate in circuit_data.gates:
            gate_type_idx = gate_registry.get_gate_type_index(gate.name)
            gate_indices.append(gate_type_idx)
            
            # 큐빗 위치 (최대 2개, 패딩 -1)
            qubits = gate.qubits[:2] if gate.qubits else [0]
            if len(qubits) == 1:
                qubits.append(-1)
            qubit_positions.append(qubits)
            
            # 파라미터 (첫 번째만 사용, 없으면 0.0)
            param = gate.parameters[0] if gate.parameters else 0.0
            gate_parameters.append(float(param))
        
        # 타겟 프로퍼티 추출
        result = circuit_data.measurement_result
        target_properties = {
            'fidelity': float(result.fidelity) if result.fidelity is not None else 0.0,
            'entanglement': float(result.entanglement) if result.entanglement is not None else 0.0,
            'robust_fidelity': float(result.robust_fidelity) if result.robust_fidelity is not None else 0.0,
        }
        
        # Expressibility 처리
        if result.expressibility and isinstance(result.expressibility, dict):
            target_properties['expressibility'] = float(result.expressibility.get('kl_divergence', 0.0))
        else:
            target_properties['expressibility'] = 0.0
        
        return {
            'gate_sequence': torch.tensor(gate_indices, dtype=torch.long),
            'qubit_positions': torch.tensor(qubit_positions, dtype=torch.long),
            'gate_parameters': torch.tensor(gate_parameters, dtype=torch.float),
            'target_properties': target_properties,
            'circuit_data': circuit_data
        }
    
    def get_target_format(self) -> str:
        return 'property'
    
    def get_collate_fn(self) -> Callable:
        return self._property_collate_fn
    
    def _property_collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """프로퍼티 예측용 배치 생성"""
        batch_size = len(batch)
        max_seq_len = max(len(item['gate_sequence']) for item in batch)
        
        # 배치 텐서 초기화
        batch_gates = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        batch_qubits = torch.full((batch_size, max_seq_len, 2), -1, dtype=torch.long)
        batch_params = torch.zeros(batch_size, max_seq_len, dtype=torch.float)
        batch_masks = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
        
        # 타겟 프로퍼티 배치
        target_tensors = {}
        property_names = ['fidelity', 'entanglement', 'expressibility', 'robust_fidelity']
        for prop_name in property_names:
            target_tensors[prop_name] = torch.tensor(
                [item['target_properties'][prop_name] for item in batch], 
                dtype=torch.float32
            )
        
        # 배치 데이터 채우기
        for i, item in enumerate(batch):
            seq_len = len(item['gate_sequence'])
            batch_gates[i, :seq_len] = item['gate_sequence']
            batch_qubits[i, :seq_len] = item['qubit_positions']
            batch_params[i, :seq_len] = item['gate_parameters']
            batch_masks[i, :seq_len] = True
        
        return {
            'input_sequence': batch_gates,
            'gate_sequence': batch_gates,
            'qubit_positions': batch_qubits,
            'gate_parameters': batch_params,
            'attention_mask': batch_masks,
            'target_properties': target_tensors,
            'circuit_specs': [item['circuit_data'] for item in batch]
        }


class DecisionTransformerCompatDataset(BaseQuantumDataset):
    """Decision Transformer 호환 데이터셋 (기존 시그니처 유지)"""
    
    def __init__(self, circuit_data: List[CircuitData], max_seq_len: int = 192, 
                 rtg_calculator=None, enable_rtg: bool = True):
        super().__init__(circuit_data)
        self.max_seq_len = max_seq_len
        self.rtg_calculator = rtg_calculator
        self.enable_rtg = enable_rtg
    
    def _get_formatted_item(self, idx: int) -> Dict[str, Any]:
        """Decision Transformer용 에피소드 포맷으로 데이터 반환"""
        circuit_data = self.circuit_data[idx]
        
        # 게이트 시퀀스를 액션으로 변환
        actions = []
        positions = []
        parameters = []
        
        from gates import gate_registry
        
        for gate in circuit_data.gates:
            # 게이트 타입 인덱스
            gate_type_idx = gate_registry.get_gate_type_index(gate.name)
            actions.append(gate_type_idx)
            
            # 큐빗 위치 [qubit1, qubit2] 형식, 단일 큐빗은 [qubit1, -1]
            qubits = gate.qubits[:2] if gate.qubits else [0]
            if len(qubits) == 1:
                qubits.append(-1)
            positions.append(qubits)
            
            # 파라미터 (첫 번째만 사용)
            param = gate.parameters[0] if gate.parameters else 0.0
            parameters.append(float(param))
        
        # 시퀀스 길이 제한 및 패딩
        seq_len = min(len(actions), self.max_seq_len)
        
        # 패딩 처리
        while len(actions) < self.max_seq_len:
            actions.append(0)  # 패딩 게이트
            positions.append([-1, -1])  # 패딩 위치
            parameters.append(0.0)  # 패딩 파라미터
        
        # 텐서 변환
        actions_tensor = torch.tensor(actions[:self.max_seq_len], dtype=torch.long)
        positions_tensor = torch.tensor(positions[:self.max_seq_len], dtype=torch.long)
        parameters_tensor = torch.tensor(parameters[:self.max_seq_len], dtype=torch.float)
        
        # 타겟 프로퍼티
        result = circuit_data.measurement_result
        target_properties = torch.tensor([
            float(result.fidelity) if result.fidelity is not None else 0.0,
            float(result.entanglement) if result.entanglement is not None else 0.0,
            float(result.expressibility.get('kl_divergence', 0.0)) if result.expressibility else 0.0
        ], dtype=torch.float)
        
        return {
            'actions': actions_tensor,
            'positions': positions_tensor,
            'parameters': parameters_tensor,
            'target_properties': target_properties,
            'seq_length': seq_len,
            'circuit_spec': circuit_data.circuit_spec
        }
    
    def get_target_format(self) -> str:
        return 'decision_transformer'
    
    def get_collate_fn(self) -> Callable:
        return self._decision_transformer_collate_fn
    
    def _decision_transformer_collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Decision Transformer용 SAR 시퀀스 배치 생성"""
        from .simple_dt_collator import SimpleDecisionTransformerCollator
        
        # 기존 collator 재사용 (호환성 유지)
        collator = SimpleDecisionTransformerCollator(d_model=512)
        
        # 배치 형식을 기존 collator가 기대하는 형식으로 변환
        formatted_batch = []
        for item in batch:
            formatted_batch.append({
                'actions': item['actions'],
                'positions': item['positions'], 
                'parameters': item['parameters'],
                'target_properties': item['target_properties'],
                'seq_length': item['seq_length']
            })
        
        return collator(formatted_batch)


class AugmentedQuantumCompatDataset(BaseQuantumDataset):
    """증강 데이터셋 호환 클래스 (기존 시그니처 유지)"""
    
    def __init__(self, base_dataset: BaseQuantumDataset, 
                 noise_samples: int = 500, 
                 param_random_samples: int = 1000):
        # 기존 AugmentedQuantumDataset 로직 재사용
        from .augmented_dataset import AugmentedQuantumDataset
        from .quantum_circuit_dataset import QuantumCircuitDataset
        
        # BaseQuantumDataset을 QuantumCircuitDataset으로 변환
        if isinstance(base_dataset, BaseQuantumDataset):
            quantum_dataset = QuantumCircuitDataset(base_dataset.circuit_data)
        else:
            quantum_dataset = base_dataset
            
        self.augmented_dataset = AugmentedQuantumDataset(
            quantum_dataset, noise_samples, param_random_samples
        )
        
        # 증강된 데이터로 초기화
        super().__init__(self.augmented_dataset.augmented_data)
    
    def _get_formatted_item(self, idx: int) -> CircuitData:
        """기본 CircuitData 반환 (기존 호환성)"""
        return self.circuit_data[idx]
    
    def get_target_format(self) -> str:
        return 'augmented'
    
    def get_collate_fn(self) -> Callable:
        # 기본 collate function 사용
        return lambda batch: batch


# 팩토리 함수들 (기존 코드와의 호환성 유지)
def create_property_dataset(circuit_data: List[CircuitData]) -> PropertyPredictionDataset:
    """프로퍼티 예측용 데이터셋 생성"""
    return PropertyPredictionDataset(circuit_data)


def create_decision_transformer_dataset(circuit_data: List[CircuitData], 
                                       max_seq_len: int = 192,
                                       rtg_calculator=None,
                                       enable_rtg: bool = True) -> DecisionTransformerCompatDataset:
    """Decision Transformer용 데이터셋 생성"""
    return DecisionTransformerCompatDataset(
        circuit_data, max_seq_len, rtg_calculator, enable_rtg
    )


def create_augmented_dataset(base_dataset: BaseQuantumDataset,
                           noise_samples: int = 500,
                           param_random_samples: int = 1000) -> AugmentedQuantumCompatDataset:
    """증강 데이터셋 생성"""
    return AugmentedQuantumCompatDataset(base_dataset, noise_samples, param_random_samples)
