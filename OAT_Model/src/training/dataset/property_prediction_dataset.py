"""
Property Prediction Dataset Module

CircuitSpec으로부터 얽힘도, fidelity, expressibility를 예측하는 
트랜스포머 모델의 데이터셋 처리 모듈
"""
import torch
from typing import Dict, List, Any
from data.quantum_circuit_dataset import QuantumCircuitDataset, CircuitData


class PropertyPredictionDataset:
    """양자 회로 특성 예측을 위한 데이터셋 래퍼"""
    
    def __init__(self, quantum_dataset: QuantumCircuitDataset):
        """
        Args:
            quantum_dataset: QuantumCircuitDataset 인스턴스
        """
        self.quantum_dataset = quantum_dataset
        
        print(f"[INIT] Property Prediction 데이터셋 초기화: {len(self.quantum_dataset)} 샘플")
    
    def __len__(self) -> int:
        return len(self.quantum_dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """CircuitData를 Property Prediction 형식으로 변환"""
        circuit_data: CircuitData = self.quantum_dataset[idx]
        
        # Check if measurement result exists
        if circuit_data.measurement_result is None:
            raise ValueError(f"No measurement result for circuit {circuit_data.circuit_id}")
            
        measurement = circuit_data.measurement_result
        
        # Validate required fields
        if measurement.fidelity is None:
            raise ValueError(f"Missing fidelity for circuit {circuit_data.circuit_id}")
        
        # Extract expressibility (KL divergence only)
        expressibility_value = 0.0
        if measurement.expressibility and isinstance(measurement.expressibility, dict):
            kl_div = measurement.expressibility.get('kl_divergence', 0.0)
            # Use KL divergence directly as expressibility
            expressibility_value = float(kl_div)
        
        targets = {
            'entanglement': float(measurement.entanglement) if measurement.entanglement is not None else 0.0,
            'fidelity': float(measurement.fidelity),
            'expressibility': float(expressibility_value)
        }
        
        # Combined target vector (3 properties only)
        targets['combined'] = torch.tensor([
            targets['entanglement'],
            targets['fidelity'], 
            targets['expressibility']
        ], dtype=torch.float32)
        
        return {
            'circuit_spec': circuit_data.circuit_spec,
            'targets': targets,
            'metadata': {
                'num_qubits': circuit_data.num_qubits,
                'num_gates': len(circuit_data.gates),
                'circuit_id': circuit_data.circuit_id,
                'depth': measurement.depth
            }
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """배치 데이터 collation"""
    # Filter out None items from batch
    valid_batch = [item for item in batch if item is not None]
    
    if not valid_batch:
        raise ValueError("[EMPTY] - No valid items in batch")
    
    circuit_specs = [item['circuit_spec'] for item in valid_batch]
    
    # 타겟 값들을 텐서로 변환
    targets = {}
    for key in ['entanglement', 'fidelity', 'expressibility']:
        targets[key] = torch.tensor([item['targets'][key] for item in valid_batch], dtype=torch.float32)
    
    targets['combined'] = torch.stack([item['targets']['combined'] for item in valid_batch])
    
    # 메타데이터
    metadata = [item['metadata'] for item in valid_batch]
    
    return {
        'circuit_specs': circuit_specs,
        'targets': targets,
        'metadata': metadata
    }


def create_datasets(data_path: str, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                   enable_augmentation: bool = True):
    """merged_data.json을 사용한 데이터셋 분할 생성 (증강 지원)"""
    from data.quantum_circuit_dataset import DatasetManager
    from typing import Tuple
    
    # Create dataset manager
    manager = DatasetManager(unified_data_path=data_path)
    
    # Split quantum datasets
    train_quantum, val_quantum, test_quantum = manager.split_dataset(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=1.0 - train_ratio - val_ratio
    )
    
    # Apply augmentation to training set if enabled
    if enable_augmentation:
        try:
            from data.augmented_dataset import create_augmented_datasets
            train_quantum, val_quantum, test_quantum = create_augmented_datasets(
                train_quantum, val_quantum, test_quantum,
                noise_samples=500,
                param_random_samples=1000
            )
        except ImportError:
            print("[WARNING] Augmentation module not available, using original datasets")
    
    # Wrap with PropertyPredictionDataset
    train_dataset = PropertyPredictionDataset(train_quantum)
    val_dataset = PropertyPredictionDataset(val_quantum)
    test_dataset = PropertyPredictionDataset(test_quantum)
    
    print(f"📊 데이터셋 분할 완료:")
    print(f"  - Train: {len(train_dataset)} 샘플")
    print(f"  - Validation: {len(val_dataset)} 샘플")
    print(f"  - Test: {len(test_dataset)} 샘플")
    
    return train_dataset, val_dataset, test_dataset
