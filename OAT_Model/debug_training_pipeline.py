#!/usr/bin/env python3
"""
Training Pipeline Debug Tool
체계적으로 데이터 흐름과 텐서 상태를 분석하는 디버깅 도구
"""

import torch
import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from data.decision_transformer_dataset import DecisionTransformerDataset
from data.quantum_circuit_dataset import DatasetManager
from rtg.core.rtg_calculator import RTGCalculator
from quantumcommon.gates import gate_registry

def debug_data_pipeline():
    """데이터 파이프라인 전체 흐름 디버깅"""
    print("=" * 60)
    print("🔍 TRAINING PIPELINE DEBUG ANALYSIS")
    print("=" * 60)
    
    # 1. Gate Registry 상태 확인
    print("\n1️⃣ GATE REGISTRY ANALYSIS")
    gate_vocab = gate_registry.get_gate_vocab()
    print(f"   Gate vocabulary size: {len(gate_vocab)}")
    print(f"   Gate types: {list(gate_vocab.keys())}")
    print(f"   Gate indices: {list(gate_vocab.values())}")
    
    # 2. 데이터셋 로딩 및 샘플 분석
    print("\n2️⃣ DATASET LOADING ANALYSIS")
    try:
        # Load sample data
        data_path = "dummy_quantum_dataset.json"
        if not os.path.exists(data_path):
            print(f"   ❌ Dataset file not found: {data_path}")
            return
            
        dataset_manager = DatasetManager()
        quantum_dataset = dataset_manager.merge_data([data_path])
        
        print(f"   ✅ Loaded {len(quantum_dataset)} circuit samples")
        
        # Sample circuit analysis
        if quantum_dataset:
            sample = quantum_dataset[0]
            circuit_spec = sample.circuit_spec
            print(f"   Sample circuit: {circuit_spec.circuit_id}")
            print(f"   Gates count: {len(circuit_spec.gates)}")
            print(f"   Qubits: {circuit_spec.num_qubits}")
            
            # Gate type analysis
            gate_types = [gate.name.lower() for gate in circuit_spec.gates]
            print(f"   Gate types in sample: {set(gate_types)}")
            
            # Check for unknown gates
            unknown_gates = [g for g in gate_types if g not in gate_vocab]
            if unknown_gates:
                print(f"   ⚠️ Unknown gates found: {set(unknown_gates)}")
            else:
                print(f"   ✅ All gates are in vocabulary")
                
    except Exception as e:
        print(f"   ❌ Dataset loading failed: {e}")
        return
    
    # 3. RTG Calculator 분석
    print("\n3️⃣ RTG CALCULATOR ANALYSIS")
    try:
        rtg_calculator = RTGCalculator()
        target_properties = {'entanglement': 0.8, 'expressibility': 0.7}
        
        rtg_values, rewards, properties = rtg_calculator.calculate_rtg(
            circuit_spec, target_properties
        )
        
        print(f"   ✅ RTG calculation successful")
        print(f"   RTG values shape: {len(rtg_values)}")
        print(f"   RTG range: [{min(rtg_values):.3f}, {max(rtg_values):.3f}]")
        print(f"   Rewards shape: {len(rewards)}")
        print(f"   Properties: {properties}")
        
    except Exception as e:
        print(f"   ❌ RTG calculation failed: {e}")
        return
    
    # 4. DecisionTransformerDataset 분석
    print("\n4️⃣ DATASET EPISODE CREATION ANALYSIS")
    try:
        dt_dataset = DecisionTransformerDataset(
            quantum_dataset=quantum_dataset,
            rtg_calculator=rtg_calculator,
            max_seq_length=64,
            target_properties=target_properties,
            d_model=256
        )
        
        print(f"   ✅ Dataset created with {len(dt_dataset)} episodes")
        
        # Analyze first episode
        if len(dt_dataset) > 0:
            episode = dt_dataset[0]
            print(f"   Episode keys: {list(episode.keys())}")
            
            # Tensor shape analysis
            for key, value in episode.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                    if key == 'actions':
                        print(f"      Actions range: [{value.min().item():.3f}, {value.max().item():.3f}]")
                        # Check for out-of-bounds values
                        if value.min() < 0 or value.max() >= len(gate_vocab):
                            print(f"      ⚠️ Actions contain out-of-bounds values!")
                            print(f"      Vocab size: {len(gate_vocab)}")
                else:
                    print(f"   {key}: {type(value)} = {value}")
                    
    except Exception as e:
        print(f"   ❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Embedding Pipeline 분석
    print("\n5️⃣ EMBEDDING PIPELINE ANALYSIS")
    try:
        from data.embedding_pipeline_refactored import EmbeddingPipeline, EmbeddingConfig
        
        embedding_config = EmbeddingConfig(
            d_model=256,
            n_gate_types=len(gate_vocab),
            n_qubits=circuit_spec.num_qubits,
            max_seq_len=64
        )
        
        embedding_pipeline = EmbeddingPipeline(embedding_config)
        result = embedding_pipeline.process_circuit(circuit_spec)
        
        print(f"   ✅ Embedding pipeline successful")
        print(f"   Result keys: {list(result.keys())}")
        
        if 'decision_transformer' in result:
            dt_result = result['decision_transformer']
            print(f"   DT result keys: {list(dt_result.keys())}")
            
            if 'input_sequence' in dt_result:
                seq = dt_result['input_sequence']
                print(f"   Input sequence shape: {seq.shape}")
                print(f"   Input sequence range: [{seq.min().item():.3f}, {seq.max().item():.3f}]")
                
    except Exception as e:
        print(f"   ❌ Embedding pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎯 DEBUGGING RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = [
        "1. 데이터 파이프라인 통합: EmbeddingPipeline → Dataset → Collator 일관성 확보",
        "2. Gate vocabulary 검증: 모든 gate가 vocab에 포함되는지 확인",
        "3. Tensor lifecycle 관리: Loss computation에서 detach() 적절히 사용",
        "4. Action encoding 표준화: Gate index, qubit positions, parameters 형식 통일",
        "5. 단계별 디버깅: 각 컴포넌트를 독립적으로 테스트"
    ]
    
    for rec in recommendations:
        print(f"   • {rec}")
    
    print(f"\n💡 다음 명령어로 이 스크립트 실행:")
    print(f"   python debug_training_pipeline.py")

if __name__ == "__main__":
    debug_data_pipeline()
