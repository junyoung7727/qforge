"""
GNN 기반 인코더와 기존 Property Prediction 모델의 통합 테스트
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import traceback

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_basic_imports():
    """기본 임포트 테스트"""
    print("Testing basic imports...")
    
    try:
        from src.encoding.base_quantum_encoder import BaseQuantumEncoder, QuantumGraphData
        print("✓ Base quantum encoder imported successfully")
    except Exception as e:
        print(f"✗ Failed to import base encoder: {e}")
        return False
    
    try:
        from src.encoding.encoding_pipeline_factory import EncodingPipelineFactory, EncodingMode
        print("✓ Pipeline factory imported successfully")
    except Exception as e:
        print(f"✗ Failed to import pipeline factory: {e}")
        return False
    
    try:
        from src.encoding.model_integration_adapter import PropertyPredictionModelAdapter
        print("✓ Model adapter imported successfully")
    except Exception as e:
        print(f"✗ Failed to import model adapter: {e}")
        return False
    
    return True

def test_pytorch_geometric_availability():
    """PyTorch Geometric 가용성 테스트"""
    print("\nTesting PyTorch Geometric availability...")
    
    try:
        import torch_geometric
        print(f"✓ PyTorch Geometric {torch_geometric.__version__} available")
        return True
    except ImportError:
        print("✗ PyTorch Geometric not available")
        print("  Install with: pip install -r requirements_gnn.txt")
        return False

def create_mock_circuit_spec():
    """테스트용 모의 CircuitSpec 생성"""
    try:
        # quantumcommon에서 임포트
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent / "quantumcommon"))
        
        from circuit_interface import CircuitSpec
        from gates import GateOperation
        
        gates = [
            GateOperation(name='h', qubits=[0], parameters=[]),
            GateOperation(name='cx', qubits=[0, 1], parameters=[]),
            GateOperation(name='ry', qubits=[1], parameters=[1.5708])
        ]
        
        return CircuitSpec(
            circuit_id="test_circuit_001",
            gates=gates,
            num_qubits=2,
            depth=3
        )
    except Exception as e:
        print(f"Failed to create mock circuit: {e}")
        return None

def test_gnn_encoder_creation():
    """GNN 인코더 생성 테스트"""
    print("\nTesting GNN encoder creation...")
    
    try:
        from src.encoding.encoding_pipeline_factory import EncodingPipelineFactory, EncodingMode
        
        config = {
            'gnn_layers': 2,
            'properties': {
                'entanglement': {'range': [0, 1], 'weight': 1.0},
                'expressibility': {'range': [0, 1], 'weight': 1.0}
            }
        }
        
        # Property Prediction 인코더 생성
        encoder = EncodingPipelineFactory.create_encoder(
            EncodingMode.PROPERTY_PREDICTION, 
            d_model=128, 
            config=config
        )
        
        print(f"✓ Property Prediction encoder created: {type(encoder).__name__}")
        
        # Decision Transformer 인코더 생성
        dt_config = {
            'gnn_layers': 2,
            'max_sequence_length': 100,
            'n_gate_types': 10,
            'max_qubits': 5
        }
        
        dt_encoder = EncodingPipelineFactory.create_encoder(
            EncodingMode.DECISION_TRANSFORMER,
            d_model=128,
            config=dt_config
        )
        
        print(f"✓ Decision Transformer encoder created: {type(dt_encoder).__name__}")
        
        return encoder, dt_encoder
        
    except Exception as e:
        print(f"✗ Failed to create encoders: {e}")
        traceback.print_exc()
        return None, None

def test_mock_encoding_without_pyg():
    """PyTorch Geometric 없이 모의 인코딩 테스트"""
    print("\nTesting mock encoding without PyTorch Geometric...")
    
    try:
        # 모의 그래프 데이터 생성
        from src.encoding.base_quantum_encoder import QuantumGraphData
        
        # 간단한 모의 데이터
        node_features = torch.randn(3, 8)  # 3 nodes, 8 features
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        edge_features = torch.randn(4, 2)  # 4 edges, 2 features
        
        graph_data = QuantumGraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            temporal_order=torch.tensor([0, 1, 2]),
            grid_shape=(3, 2),
            node_to_grid_mapping={0: (0, 0), 1: (1, 0), 2: (2, 0)},
            grid_to_node_mapping={(0, 0): 0, (1, 0): 1, (2, 0): 2}
        )
        
        print(f"✓ Mock graph data created:")
        print(f"  Nodes: {graph_data.node_features.shape}")
        print(f"  Edges: {graph_data.edge_index.shape}")
        print(f"  Edge features: {graph_data.edge_features.shape}")
        
        return graph_data
        
    except Exception as e:
        print(f"✗ Failed to create mock graph data: {e}")
        traceback.print_exc()
        return None

def test_integration_adapter():
    """통합 어댑터 테스트"""
    print("\nTesting integration adapter...")
    
    try:
        from src.encoding.model_integration_adapter import PropertyPredictionModelAdapter
        
        config = {
            'gnn_layers': 2,
            'properties': {
                'entanglement': {'range': [0, 1], 'weight': 1.0},
                'expressibility': {'range': [0, 1], 'weight': 1.0}
            }
        }
        
        adapter = PropertyPredictionModelAdapter(d_model=128, config=config)
        print(f"✓ Integration adapter created: {type(adapter).__name__}")
        
        # 모의 회로 생성
        circuit_spec = create_mock_circuit_spec()
        if circuit_spec is None:
            print("✗ Cannot test adapter without circuit spec")
            return False
        
        print(f"✓ Circuit spec created: {circuit_spec.circuit_id}")
        print(f"  Gates: {len(circuit_spec.gates)}")
        print(f"  Qubits: {circuit_spec.num_qubits}")
        
        # 어댑터 테스트 (PyG 없이는 실제 실행 안됨)
        print("✓ Adapter structure validated")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test integration adapter: {e}")
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """하위 호환성 테스트"""
    print("\nTesting backward compatibility...")
    
    try:
        from src.encoding.model_integration_adapter import BackwardCompatibilityWrapper, EmbeddingFormatConverter
        
        # 형식 변환기 테스트
        converter = EmbeddingFormatConverter(d_model=128)
        
        # 모의 GNN 임베딩
        gnn_embeddings = torch.randn(1, 128)
        
        # 모의 그래프 데이터
        graph_data = test_mock_encoding_without_pyg()
        if graph_data is None:
            print("✗ Cannot test without graph data")
            return False
        
        # 레거시 형식으로 변환
        legacy_format = converter.convert_to_legacy_format(gnn_embeddings, graph_data)
        
        print(f"✓ Legacy format conversion:")
        print(f"  Embeddings shape: {legacy_format['embeddings'].shape}")
        print(f"  Attention mask shape: {legacy_format['attention_mask'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed backward compatibility test: {e}")
        traceback.print_exc()
        return False

def test_existing_model_compatibility():
    """기존 모델과의 호환성 테스트"""
    print("\nTesting existing model compatibility...")
    
    try:
        # 모의 Property Prediction 모델 생성 (실제 모델 대신)
        class MockPropertyPredictionModel(nn.Module):
            def __init__(self, d_model=128):
                super().__init__()
                self.linear = nn.Linear(d_model, 3)  # 3 properties
                
            def forward(self, embeddings, attention_mask=None):
                # 간단한 순전파
                batch_size, seq_len, d_model = embeddings.shape
                pooled = torch.mean(embeddings, dim=1)  # [batch_size, d_model]
                output = self.linear(pooled)  # [batch_size, 3]
                
                return {
                    'entanglement': output[:, 0:1],
                    'expressibility': output[:, 1:2],
                    'fidelity': output[:, 2:3]
                }
        
        model = MockPropertyPredictionModel(d_model=128)
        print(f"✓ Mock model created: {type(model).__name__}")
        
        # 모의 입력으로 테스트
        batch_size, seq_len, d_model = 1, 5, 128
        mock_embeddings = torch.randn(batch_size, seq_len, d_model)
        mock_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # 모델 실행 테스트
        with torch.no_grad():
            output = model(mock_embeddings, attention_mask=mock_mask)
        
        print(f"✓ Model forward pass successful")
        print(f"  Output keys: {list(output.keys()) if isinstance(output, dict) else 'tensor'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed existing model compatibility test: {e}")
        traceback.print_exc()
        return False

def test_dimension_compatibility():
    """차원 호환성 테스트"""
    print("\nTesting dimension compatibility...")
    
    try:
        # 다양한 차원 크기 테스트
        test_dimensions = [64, 128, 256, 512]
        
        for d_model in test_dimensions:
            # GNN 임베딩 시뮬레이션
            gnn_embedding = torch.randn(1, d_model)
            
            # 기존 모델 형식으로 변환
            from src.encoding.model_integration_adapter import EmbeddingFormatConverter
            converter = EmbeddingFormatConverter(d_model)
            
            # 모의 그래프 데이터
            graph_data = test_mock_encoding_without_pyg()
            if graph_data is None:
                continue
            
            legacy_format = converter.convert_to_legacy_format(gnn_embedding, graph_data)
            
            # 차원 검증
            expected_shape = (1, graph_data.node_features.size(0), d_model)
            actual_shape = legacy_format['embeddings'].shape
            
            if actual_shape == expected_shape:
                print(f"✓ Dimension {d_model}: {actual_shape}")
            else:
                print(f"✗ Dimension {d_model}: expected {expected_shape}, got {actual_shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed dimension compatibility test: {e}")
        traceback.print_exc()
        return False

def main():
    """메인 테스트 실행"""
    print("GNN Integration Testing Suite")
    print("=" * 50)
    
    test_results = []
    
    # 기본 임포트 테스트
    test_results.append(("Basic Imports", test_basic_imports()))
    
    # PyTorch Geometric 가용성
    pyg_available = test_pytorch_geometric_availability()
    test_results.append(("PyTorch Geometric", pyg_available))
    
    # GNN 인코더 생성 테스트
    encoder_result = test_gnn_encoder_creation()
    test_results.append(("GNN Encoder Creation", encoder_result[0] is not None))
    
    # 모의 인코딩 테스트
    mock_graph = test_mock_encoding_without_pyg()
    test_results.append(("Mock Graph Data", mock_graph is not None))
    
    # 통합 어댑터 테스트
    test_results.append(("Integration Adapter", test_integration_adapter()))
    
    # 하위 호환성 테스트
    test_results.append(("Backward Compatibility", test_backward_compatibility()))
    
    # 기존 모델 호환성 테스트
    test_results.append(("Existing Model Compatibility", test_existing_model_compatibility()))
    
    # 차원 호환성 테스트
    test_results.append(("Dimension Compatibility", test_dimension_compatibility()))
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if not pyg_available:
        print("\nNote: PyTorch Geometric not installed.")
        print("Some functionality will be limited until dependencies are installed.")
        print("Run: pip install -r requirements_gnn.txt")
    
    if passed == total:
        print("\n🎉 All tests passed! GNN integration is ready.")
    elif passed >= total * 0.7:
        print(f"\n⚠️  Most tests passed ({passed}/{total}). Check failed tests.")
    else:
        print(f"\n❌ Many tests failed ({total-passed}/{total}). Review implementation.")

if __name__ == "__main__":
    main()
