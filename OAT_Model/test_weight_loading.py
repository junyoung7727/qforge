#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

import torch
from src.rtg.model_loader import load_property_predictor
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent.parent.parent / "quantumcommon"))
from circuit_interface import CircuitSpec
from gates import GateOperation

def test_weight_loading():
    """가중치 로딩 상태 확인"""
    print("=== 가중치 로딩 테스트 ===")
    
    # 1. 모델 로드

    model = load_property_predictor(r'C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\weights\best_model.pt')
    print(f"✅ 모델 로드 성공: {type(model).__name__}")
    print(f"📱 디바이스: {getattr(model, 'device', 'N/A')}")


    # 2. 가중치 통계 확인 (랜덤 초기화가 아닌지 확인)
    print("\n=== 가중치 통계 ===")
    weight_stats = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.numel() > 100:  # 큰 가중치만 확인
            mean_val = param.mean().item()
            std_val = param.std().item()
            weight_stats.append((name, mean_val, std_val))
            print(f"{name}: mean={mean_val:.6f}, std={std_val:.6f}")
            if len(weight_stats) >= 5:  # 처음 5개만 출력
                break
    
    # 3. 더미 입력으로 forward pass 테스트
    print("\n=== Forward Pass 테스트 ===")
    try:
        dummy_spec =  CircuitSpec(
            num_qubits=4,
            gates=[
                GateOperation(name="H", qubits=[0]),
                GateOperation(name="H", qubits=[1]),
                GateOperation(name="CNOT", qubits=[0, 2]),
                GateOperation(name="CNOT", qubits=[1, 3]),
                GateOperation(name="RZ", qubits=[2], parameters=[0.5]),
                GateOperation(name="RY", qubits=[3], parameters=[0.3])
            ],
            circuit_id="complex_circuit",
            depth=6
        )
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_spec)
            
        if isinstance(output, dict):
            print("✅ Forward pass 성공")
            for k, v in output.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: {v.shape}, 값={v.squeeze().item() if v.numel() == 1 else 'tensor'}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"⚠️ 예상과 다른 출력 타입: {type(output)}")
            
    except Exception as e:
        print(f"❌ Forward pass 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 가중치가 실제로 로드되었는지 확인 (체크포인트와 비교)
    print("\n=== 체크포인트 가중치 비교 ===")
    try:
        checkpoint = torch.load('weights/best_model.pt', map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
        
        # 몇 개 레이어의 가중치가 일치하는지 확인
        matches = 0
        total_checked = 0
        for name, param in model.named_parameters():
            if name in state_dict and total_checked < 3:
                checkpoint_param = state_dict[name]
                if torch.allclose(param.cpu(), checkpoint_param, atol=1e-6):
                    matches += 1
                    print(f"✅ {name}: 가중치 일치")
                else:
                    print(f"❌ {name}: 가중치 불일치")
                total_checked += 1
                
        print(f"가중치 일치율: {matches}/{total_checked}")
        
    except Exception as e:
        print(f"⚠️ 체크포인트 비교 실패: {e}")
    
    return True

if __name__ == "__main__":
    test_weight_loading()
