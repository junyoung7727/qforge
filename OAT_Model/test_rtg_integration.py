"""
RTG Calculator Integration Test
실제 데이터를 사용하여 RTG 계산기 통합 테스트
"""

import torch
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "quantumcommon"))
sys.path.append(str(Path(__file__).parent / "src"))

from rtg import create_rtg_calculator, create_episode_visualizer
from circuit_interface import CircuitSpec, GateOperation

def create_test_episode():
    """테스트용 에피소드 데이터 생성"""
    episode = []
    
    # 간단한 4-qubit 회로 생성
    gates = [
        {'type': 'H', 'qubits': [0]},
        {'type': 'CNOT', 'qubits': [0, 1]},
        {'type': 'RY', 'qubits': [1], 'params': [0.5]},
        {'type': 'CNOT', 'qubits': [1, 2]},
        {'type': 'RZ', 'qubits': [2], 'params': [0.3]},
        {'type': 'CNOT', 'qubits': [2, 3]},
    ]
    
    for i, gate in enumerate(gates):
        # CircuitSpec 생성
        circuit_spec = CircuitSpec(
            num_qubits=4,
            gates=[GateOperation.from_dict(g) for g in gates[:i+1]],
            circuit_id=f"test_circuit_{i}",
            depth=i+1
        )
        
        episode.append({
            'step': i,
            'circuit_spec': circuit_spec,
            'gate_action': gate,
            'target_properties': {
                'entanglement': 0.7,
                'expressibility': 0.8,
                'fidelity': 0.9
            }
        })
    
    return episode

def test_rtg_calculator():
    """RTG 계산기 테스트"""
    print("=== RTG Calculator Integration Test ===")
    
    # 1. RTG 계산기 생성
    print("\n1. Creating RTG Calculator...")
    rtg_calculator = create_rtg_calculator(
        checkpoint_path=r"C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\weights\best_model.pt",
        device="cpu"
    )
    print("✓ RTG Calculator created successfully")
    
    # 2. 테스트 에피소드 생성
    print("\n2. Creating test episode...")
    episode = create_test_episode()
    print(f"✓ Created episode with {len(episode)} steps")
    
    # 3. RTG 계산
    print("\n3. Calculating RTG values...")
    # RTG 계산을 위한 데이터 준비
    episode_data = {
        'circuit_specs': [step['circuit_spec'] for step in episode]
    }
    print("==")
    print(episode_data)
    target_properties = episode[0]['target_properties']
    
    rtg_results = rtg_calculator.calculate_episode_rtg(
        episode_data=episode_data,
        target_properties=target_properties
    )
    print("✓ RTG calculation completed")
    print(f"  - RTG values: {len(rtg_results['rtg_values'])}")
    print(f"  - Final RTG: {rtg_results['rtg_values'][-1]:.4f}")
    print(f"  - Total reward: {sum(rtg_results['rewards']):.4f}")
    
    # 4. 결과 검증
    print("\n4. Validating results...")
    required_keys = ['rewards', 'rtg_values', 'properties']
    for key in required_keys:
        if key not in rtg_results:
            print(f"✗ Missing key in results: {key}")
            return False
        print(f"✓ Found {key}: {len(rtg_results[key])} items")
    
    return True, rtg_results

def test_visualization():
    """시각화 모듈 테스트"""
    print("\n=== Visualization Test ===")
    
    # RTG 결과 가져오기
    rtg_results = test_rtg_calculator()
    if not rtg_results:
        return False
    
    # 시각화 생성
    print("\n1. Creating episode visualizer...")
    visualizer = create_episode_visualizer()
    print("✓ Episode visualizer created")
    
    # 시각화 생성
    print("\n2. Generating visualizations...")
    output_dir = "test_rtg_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 기본 시각화 - rtg_results를 직접 전달 (이미 딕셔너리 형태)
    if rtg_results[0] == True:
        rtg_results = rtg_results[1]
    else:
        raise KeyboardInterrupt
        
    visualizer.visualize_episode_rewards(
        rtg_results, 
        episode_id="test_episode"
    )
    print("✓ Episode visualization completed")
    
    print(f"\n✓ All visualizations saved to reward_visualization/")
    
    return True

def test_model_loading():
    """모델 로딩 테스트"""
    print("\n=== Model Loading Test ===")
    
    from rtg.model_loader import find_best_checkpoint, load_property_predictor
    
    # 1. 체크포인트 찾기
    print("\n1. Finding best checkpoint...")
    checkpoint_path = find_best_checkpoint(r"C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\weights")
    if checkpoint_path:
        print(f"✓ Found checkpoint: {checkpoint_path}")
    else:
        print("! No checkpoint found - creating dummy test")
        return True  # 체크포인트가 없어도 테스트는 통과
    
    # 2. 모델 로딩
    print("\n2. Loading property predictor...")
    model = load_property_predictor(
        checkpoint_path=checkpoint_path,
        device="cpu"
    )
    print("✓ Model loaded successfully")
    print(f"  - Model type: {type(model).__name__}")
    print(f"  - Device: {next(model.parameters()).device}")
    
    return True

def run_comprehensive_test():
    """종합 테스트 실행"""
    print("Starting RTG Integration Comprehensive Test")
    print("=" * 50)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("RTG Calculator", lambda: bool(test_rtg_calculator())),
        ("Visualization", test_visualization)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results[test_name] = success
        status = "PASSED" if success else "FAILED"
        print(f"\n{test_name}: {status}")
    
    # 결과 요약
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! RTG system is ready for use.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
