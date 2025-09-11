"""
GNN 기반 양자회로 인코더 사용 예제
새로운 GNN 인코딩 파이프라인 사용법과 기존 모델과의 통합 방법
"""

import torch
import sys
from pathlib import Path
import pathlib

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.encoding.encoding_pipeline_factory import EncodingPipelineFactory, EncodingMode
from src.encoding.model_integration_adapter import PropertyPredictionModelAdapter, HybridTrainingManager
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent / "quantumcommon"))
from gates import gate_registry, GateType, GateOperation
from src.data.quantum_circuit_dataset import CircuitSpec, GateOperation

def create_sample_circuit():
    """샘플 양자회로 생성"""
    gates = [
        GateOperation(gate_type='h', qubits=[0], parameters=[]),
        GateOperation(gate_type='cx', qubits=[0, 1], parameters=[]),
        GateOperation(gate_type='ry', qubits=[1], parameters=[1.5708]),
        GateOperation(gate_type='cx', qubits=[1, 2], parameters=[]),
        GateOperation(gate_type='rz', qubits=[2], parameters=[0.7854])
    ]
    
    return CircuitSpec(
        circuit_id="example_circuit_001",
        gates=gates,
        num_qubits=3,
        properties={
            'entanglement': 0.75,
            'expressibility': 0.82,
            'fidelity': 0.95
        }
    )

def example_property_prediction_encoding():
    """Property Prediction을 위한 GNN 인코딩 예제"""
    print("=== Property Prediction GNN Encoding Example ===")
    
    # 설정
    d_model = 256
    config = {
        'gnn_layers': 3,
        'properties': {
            'entanglement': {'range': [0, 1], 'weight': 1.0},
            'expressibility': {'range': [0, 1], 'weight': 1.0},
            'fidelity': {'range': [0, 1], 'weight': 1.0}
        },
        'enable_caching': True,
        'batch_size': 32
    }
    
    # 파이프라인 생성
    pipeline = EncodingPipelineFactory.create_pipeline(
        EncodingMode.PROPERTY_PREDICTION, 
        d_model, 
        config
    )
    
    # 샘플 회로 생성
    circuit = create_sample_circuit()
    
    # 인코딩 수행
    try:
        result = pipeline.encode_single(circuit)
        
        print(f"Embedding shape: {result.embeddings.shape}")
        print(f"Encoding type: {result.auxiliary_data.get('encoding_type', 'unknown')}")
        
        # 속성 예측 결과
        if 'property_predictions' in result.auxiliary_data:
            predictions = result.auxiliary_data['property_predictions']
            print("Property Predictions:")
            for prop_name, pred_value in predictions.items():
                actual_value = circuit.properties.get(prop_name, 'N/A')
                print(f"  {prop_name}: predicted={pred_value.item():.4f}, actual={actual_value}")
        
        return result
        
    except Exception as e:
        print(f"Error during encoding: {e}")
        print("Note: This example requires PyTorch Geometric installation")
        return None

def example_decision_transformer_encoding():
    """Decision Transformer를 위한 증분적 인코딩 예제"""
    print("\n=== Decision Transformer Incremental Encoding Example ===")
    
    # 설정
    d_model = 256
    config = {
        'gnn_layers': 3,
        'max_sequence_length': 1000,
        'n_gate_types': 20,
        'max_qubits': 20,
        'enable_caching': True
    }
    
    # 파이프라인 생성
    pipeline = EncodingPipelineFactory.create_pipeline(
        EncodingMode.DECISION_TRANSFORMER, 
        d_model, 
        config
    )
    
    # 샘플 회로 생성
    circuit = create_sample_circuit()
    
    try:
        # 증분적 시퀀스 인코딩
        sequence_results = pipeline.encode_incremental_sequence(circuit, max_steps=3)
        
        print(f"Sequence length: {len(sequence_results)}")
        
        for i, result in enumerate(sequence_results):
            print(f"Step {i}:")
            print(f"  Embedding shape: {result.embeddings.shape}")
            
            # 액션 예측 결과
            if 'action_predictions' in result.auxiliary_data:
                action_preds = result.auxiliary_data['action_predictions']
                print(f"  Action predictions available: {list(action_preds.keys())}")
        
        return sequence_results
        
    except Exception as e:
        print(f"Error during incremental encoding: {e}")
        print("Note: This example requires PyTorch Geometric installation")
        return None

def example_hybrid_mode():
    """하이브리드 모드 사용 예제"""
    print("\n=== Hybrid Mode Example ===")
    
    # 설정
    d_model = 256
    config = {
        'gnn_layers': 3,
        'properties': {
            'entanglement': {'range': [0, 1], 'weight': 1.0},
            'expressibility': {'range': [0, 1], 'weight': 1.0}
        },
        'max_sequence_length': 1000,
        'n_gate_types': 20,
        'enable_caching': True
    }
    
    # 하이브리드 파이프라인 생성
    pipeline = EncodingPipelineFactory.create_pipeline(
        EncodingMode.HYBRID, 
        d_model, 
        config
    )
    
    # 샘플 회로 생성
    circuit = create_sample_circuit()
    
    try:
        # Property Prediction 태스크
        pp_result = pipeline.encode_single(circuit, task_type='property_prediction')
        print(f"Property Prediction - Embedding shape: {pp_result.embeddings.shape}")
        
        # Decision Transformer 태스크
        dt_result = pipeline.encode_single(circuit, task_type='decision_transformer', current_step=2)
        print(f"Decision Transformer - Embedding shape: {dt_result.embeddings.shape}")
        
        return pp_result, dt_result
        
    except Exception as e:
        print(f"Error during hybrid encoding: {e}")
        print("Note: This example requires PyTorch Geometric installation")
        return None

def example_model_integration():
    """기존 Property Prediction 모델과의 통합 예제"""
    print("\n=== Model Integration Example ===")
    
    # 설정
    d_model = 256
    config = {
        'gnn_layers': 3,
        'properties': {
            'entanglement': {'range': [0, 1], 'weight': 1.0},
            'expressibility': {'range': [0, 1], 'weight': 1.0},
            'fidelity': {'range': [0, 1], 'weight': 1.0}
        },
        'hybrid_mode': True,
        'legacy_weight': 0.3
    }
    
    try:
        # 통합 어댑터 생성
        adapter = PropertyPredictionModelAdapter(d_model, config)
        
        # 샘플 회로 생성
        circuit = create_sample_circuit()
        
        # GNN 기반 인코딩
        gnn_result = adapter.forward(circuit, use_gnn=True)
        print(f"GNN-based encoding - Embedding shape: {gnn_result['embeddings'].shape}")
        
        # 속성 예측 결과
        for prop_name in ['entanglement', 'expressibility', 'fidelity']:
            if prop_name in gnn_result:
                pred_value = gnn_result[prop_name]
                actual_value = circuit.properties.get(prop_name, 'N/A')
                print(f"  {prop_name}: predicted={pred_value.item():.4f}, actual={actual_value}")
        
        return gnn_result
        
    except Exception as e:
        print(f"Error during model integration: {e}")
        print("Note: This example requires PyTorch Geometric installation")
        return None

def example_batch_processing():
    """배치 처리 예제"""
    print("\n=== Batch Processing Example ===")
    
    # 여러 샘플 회로 생성
    circuits = []
    for i in range(5):
        gates = [
            GateOperation(gate_type='h', qubits=[0], parameters=[]),
            GateOperation(gate_type='cx', qubits=[0, 1], parameters=[]),
            GateOperation(gate_type='ry', qubits=[1], parameters=[1.5708 + i * 0.1])
        ]
        
        circuit = CircuitSpec(
            circuit_id=f"batch_circuit_{i:03d}",
            gates=gates,
            num_qubits=2,
            properties={
                'entanglement': 0.5 + i * 0.1,
                'expressibility': 0.6 + i * 0.05
            }
        )
        circuits.append(circuit)
    
    # 설정
    d_model = 256
    config = {
        'gnn_layers': 2,
        'properties': {
            'entanglement': {'range': [0, 1], 'weight': 1.0},
            'expressibility': {'range': [0, 1], 'weight': 1.0}
        },
        'batch_size': 3,
        'enable_caching': True
    }
    
    # 파이프라인 생성
    pipeline = EncodingPipelineFactory.create_pipeline(
        EncodingMode.PROPERTY_PREDICTION, 
        d_model, 
        config
    )
    
    try:
        # 배치 인코딩
        batch_results = pipeline.encode_batch(circuits)
        print(f"Processed {len(batch_results)} circuits")
        
        # 속성 트렌드 분석
        trend_analysis = pipeline.analyze_property_trends(circuits)
        print("Property Statistics:")
        for prop_name, stats in trend_analysis['statistics'].items():
            print(f"  {prop_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        
        # 캐시 통계
        cache_stats = pipeline.get_cache_stats()
        print(f"Cache statistics: {cache_stats}")
        
        return batch_results, trend_analysis
        
    except Exception as e:
        print(f"Error during batch processing: {e}")
        print("Note: This example requires PyTorch Geometric installation")
        return None

def main():
    """메인 실행 함수"""
    print("GNN-based Quantum Circuit Encoder Usage Examples")
    print("=" * 60)
    
    # 의존성 체크
    try:
        import torch_geometric
        print(f"PyTorch Geometric version: {torch_geometric.__version__}")
        dependencies_available = True
    except ImportError:
        print("Warning: PyTorch Geometric not installed. Examples will show structure but may fail.")
        print("Install with: pip install -r requirements_gnn.txt")
        dependencies_available = False
    
    print()
    
    # 예제 실행
    example_property_prediction_encoding()
    example_decision_transformer_encoding()
    example_hybrid_mode()
    example_model_integration()
    example_batch_processing()
    
    print("\n" + "=" * 60)
    if dependencies_available:
        print("All examples completed successfully!")
    else:
        print("Examples completed with dependency warnings.")
        print("Install PyTorch Geometric for full functionality.")

if __name__ == "__main__":
    main()
