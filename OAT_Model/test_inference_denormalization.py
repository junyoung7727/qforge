#!/usr/bin/env python3

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent.parent / "quantumcommon"))

import torch
from circuit_interface import CircuitSpec
from gates import GateOperation
from rtg.model_loader import load_property_predictor

def test_inference_denormalization():
    """추론 시 자동 denormalization 테스트"""
    print("=== 추론 시 자동 Denormalization 테스트 ===")
    
    # 1. 체크포인트에서 모델 로드
    model_path = r'C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\weights\best_model.pt'
    model = load_property_predictor(model_path)
    
    print(f"로드된 모델의 저장된 통계:")
    print(f"  exp_mean: {model.prediction_head.exp_mean.item():.6f}")
    print(f"  exp_std: {model.prediction_head.exp_std.item():.6f}")
    print(f"  exp_count: {model.prediction_head.exp_count.item()}")
    print(f"  ent_mean: {model.prediction_head.ent_mean.item():.6f}")
    print(f"  ent_std: {model.prediction_head.ent_std.item():.6f}")
    print(f"  fid_mean: {model.prediction_head.fid_mean.item():.6f}")
    print(f"  fid_std: {model.prediction_head.fid_std.item():.6f}")
    
    # 2. 테스트 회로 생성
    test_spec = CircuitSpec(
        num_qubits=4,
        gates=[
            GateOperation(name="H", qubits=[0]),
            GateOperation(name="H", qubits=[1]),
            GateOperation(name="CNOT", qubits=[0, 2]),
            GateOperation(name="CNOT", qubits=[1, 3]),
            GateOperation(name="RZ", qubits=[2], parameters=[0.5]),
            GateOperation(name="RY", qubits=[3], parameters=[0.3])
        ],
        circuit_id="test_circuit",
        depth=6
    )
    
    # 3. 훈련 모드에서 예측 (z-normalized 출력)
    print(f"\n=== 훈련 모드 (z-normalized 출력) ===")
    model.train()
    with torch.no_grad():
        train_outputs = model(test_spec)
        
    for prop, value in train_outputs.items():
        print(f"  {prop}: {value.squeeze().item():.6f} (z-score)")
    
    # 4. 추론 모드에서 예측 (자동 denormalized 출력)
    print(f"\n=== 추론 모드 (자동 denormalized 출력) ===")
    model.eval()
    with torch.no_grad():
        inference_outputs = model(test_spec)
        
    for prop, value in inference_outputs.items():
        print(f"  {prop}: {value.squeeze().item():.6f} (실제 값)")
    
    # 5. 수동 denormalization과 비교 검증
    print(f"\n=== 수동 Denormalization 비교 ===")
    with torch.no_grad():
        # 훈련 모드 출력을 수동으로 denormalize
        manual_denorm = {}
        manual_denorm['entanglement'] = (train_outputs['entanglement'] * model.prediction_head.ent_std + 
                                       model.prediction_head.ent_mean).squeeze().item()
        manual_denorm['fidelity'] = (train_outputs['fidelity'] * model.prediction_head.fid_std + 
                                   model.prediction_head.fid_mean).squeeze().item()
        manual_denorm['expressibility'] = (train_outputs['expressibility'] * model.prediction_head.exp_std + 
                                         model.prediction_head.exp_mean).squeeze().item()
    
    print("수동 denormalization 결과:")
    for prop, value in manual_denorm.items():
        auto_value = inference_outputs[prop].squeeze().item()
        diff = abs(value - auto_value)
        print(f"  {prop}: {value:.6f} (수동) vs {auto_value:.6f} (자동), 차이: {diff:.8f}")
        
        if diff < 1e-6:
            print(f"    ✅ 일치")
        else:
            print(f"    ❌ 불일치")
    
    # 6. 실제 expressibility 범위 확인
    print(f"\n=== Expressibility 값 범위 분석 ===")
    exp_value = inference_outputs['expressibility'].squeeze().item()
    if model.prediction_head.exp_std.item() > 1e-6:  # 유효한 통계가 있는 경우
        print(f"  예측된 expressibility: {exp_value:.6f}")
        print(f"  저장된 평균: {model.prediction_head.exp_mean.item():.6f}")
        print(f"  저장된 표준편차: {model.prediction_head.exp_std.item():.6f}")
        
        # Z-score 계산
        z_score = (exp_value - model.prediction_head.exp_mean.item()) / model.prediction_head.exp_std.item()
        print(f"  Z-score: {z_score:.3f}")
    else:
        print(f"  ⚠️ 저장된 expressibility 통계가 유효하지 않음 (std={model.prediction_head.exp_std.item():.6f})")
    
    return True

if __name__ == "__main__":
    test_inference_denormalization()
