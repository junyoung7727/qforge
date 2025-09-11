#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent.parent / "quantumcommon"))

import torch
from circuit_interface import CircuitSpec
from gates import GateOperation
from rtg.model_loader import load_property_predictor

def test_statistics_validation():
    """통계 검증 및 경고 메시지 테스트"""
    print("=== 통계 검증 테스트 ===")
    
    # 1. 체크포인트에서 모델 로드
    model_path = r'C:\Users\jungh\Documents\GitHub\Kaist\OAT_Model\weights\best_model.pt'
    model = load_property_predictor(model_path)
    
    print(f"로드된 모델의 통계:")
    print(f"  exp_mean: {model.prediction_head.exp_mean.item():.6f}")
    print(f"  exp_std: {model.prediction_head.exp_std.item():.6f}")
    print(f"  exp_count: {model.prediction_head.exp_count.item()}")
    print(f"  ent_mean: {model.prediction_head.ent_mean.item():.6f}")
    print(f"  ent_std: {model.prediction_head.ent_std.item():.6f}")
    print(f"  ent_count: {model.prediction_head.ent_count.item()}")
    print(f"  fid_mean: {model.prediction_head.fid_mean.item():.6f}")
    print(f"  fid_std: {model.prediction_head.fid_std.item():.6f}")
    print(f"  fid_count: {model.prediction_head.fid_count.item()}")
    
    # 2. 통계 유효성 검증
    print(f"\n=== 통계 유효성 검증 ===")
    stats_valid = model.prediction_head._validate_statistics()
    print(f"통계 유효성: {'✅ 유효' if stats_valid else '❌ 무효'}")
    
    # 3. 테스트 회로로 추론 수행
    print(f"\n=== 추론 테스트 (검증 포함) ===")
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
    
    # 추론 모드에서 예측 (자동 검증 및 경고 메시지 확인)
    model.eval()
    with torch.no_grad():
        outputs = model(test_spec)
        
    print(f"예측 결과:")
    for prop, value in outputs.items():
        print(f"  {prop}: {value.squeeze().item():.6f}")
    
    # 4. 수동으로 각 검증 조건 확인
    print(f"\n=== 상세 검증 결과 ===")
    pred_head = model.prediction_head
    
    # 최소 샘플 수 확인
    min_samples_ok = (
        pred_head.ent_count.item() >= pred_head.min_samples_for_stats and
        pred_head.fid_count.item() >= pred_head.min_samples_for_stats and
        pred_head.exp_count.item() >= pred_head.min_samples_for_stats
    )
    print(f"최소 샘플 수 조건: {'✅' if min_samples_ok else '❌'}")
    print(f"  - ent_count: {pred_head.ent_count.item()} >= {pred_head.min_samples_for_stats}")
    print(f"  - fid_count: {pred_head.fid_count.item()} >= {pred_head.min_samples_for_stats}")
    print(f"  - exp_count: {pred_head.exp_count.item()} >= {pred_head.min_samples_for_stats}")
    
    # 표준편차 유효성 확인
    std_valid = (
        pred_head.ent_std.item() > 1e-6 and pred_head.ent_std.item() != 1.0 and
        pred_head.fid_std.item() > 1e-6 and pred_head.fid_std.item() != 1.0 and
        pred_head.exp_std.item() > 1e-6 and pred_head.exp_std.item() != 1.0
    )
    print(f"표준편차 유효성: {'✅' if std_valid else '❌'}")
    print(f"  - ent_std: {pred_head.ent_std.item():.6f} (≠1.0: {pred_head.ent_std.item() != 1.0})")
    print(f"  - fid_std: {pred_head.fid_std.item():.6f} (≠1.0: {pred_head.fid_std.item() != 1.0})")
    print(f"  - exp_std: {pred_head.exp_std.item():.6f} (≠1.0: {pred_head.exp_std.item() != 1.0})")
    
    # 평균 유효성 확인
    mean_valid = not (
        abs(pred_head.ent_mean.item()) < 1e-8 and
        abs(pred_head.fid_mean.item()) < 1e-8 and
        abs(pred_head.exp_mean.item()) < 1e-8
    )
    print(f"평균 유효성: {'✅' if mean_valid else '❌'}")
    print(f"  - ent_mean: {pred_head.ent_mean.item():.6f}")
    print(f"  - fid_mean: {pred_head.fid_mean.item():.6f}")
    print(f"  - exp_mean: {pred_head.exp_mean.item():.6f}")
    
    return True

if __name__ == "__main__":
    test_statistics_validation()
