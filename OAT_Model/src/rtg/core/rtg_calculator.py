"""
Modular RTG (Return-To-Go) Calculator
효율적이고 모듈화된 RTG 계산 시스템
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Union
from collections import deque
from abc import ABC, abstractmethod
from pathlib import Path
import sys
import math
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))
# Removed unused import: EncodingPipelineFactory (legacy graph-based encoding)

# Add project paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from rtg.model_loader import load_property_predictor, find_best_checkpoint


class PropertyPredictor(ABC):
    """Property 예측을 위한 추상 클래스"""
    
    @abstractmethod
    def predict(self, circuit_spec) -> Dict[str, float]:
        """회로 스펙으로부터 속성 예측"""
        pass


class ModelBasedPropertyPredictor(PropertyPredictor):
    """학습된 모델 기반 Property 예측기"""
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "auto", loaded_model = None):
        """
        Args:
            checkpoint_path: 체크포인트 경로 (None이면 자동 검색)
            device: 디바이스 설정
            loaded_model: 이미 로드된 모델 (있는 경우 사용)
        """
        if loaded_model is not None:
            self.model = loaded_model
            print(f"✅ 이미 로드된 Property prediction 모델을 사용합니다")
        else:
            if checkpoint_path is None:
                checkpoint_path = find_best_checkpoint()
                if checkpoint_path is None:
                    raise FileNotFoundError("Property predictor 체크포인트를 찾을 수 없습니다")
            
            self.model = load_property_predictor(checkpoint_path, device)
        
        self.device = self.model.device if hasattr(self.model, 'device') else torch.device(device)
        
    def predict(self, circuit_spec) -> Dict[str, float]:
        """현재 아키텍처 기반 회로 스펙으로부터 속성 예측"""
        with torch.no_grad():
            # 현재 모델 인터페이스에 맞게 수정
            if hasattr(circuit_spec, 'to'):
                circuit_spec = circuit_spec.to(self.device)
            
            # 모델 forward 호출 (eval 모드에서 자동으로 denormalization됨)
            self.model.eval()  # 추론 모드 설정

            outputs = self.model(circuit_spec)

            
            # 현재 아키텍처 출력 처리 (Dict[str, torch.Tensor])
            if isinstance(outputs, dict):
                if 'expressibility' in outputs and 'expressibility' in outputs and 'expressibility' in outputs:
                    entanglement = outputs['entanglement'].squeeze().item() 
                    fidelity = outputs['fidelity'].squeeze().item() 
                    expressibility = outputs['expressibility'].squeeze().item()

            # NaN/Inf 검증 및 처리
            if torch.isnan(torch.tensor(entanglement)) or torch.isinf(torch.tensor(entanglement)):
                print("⚠️ entanglement에서 NaN/Inf 감지, 0으로 대체")
                entanglement = 0.0
            if torch.isnan(torch.tensor(fidelity)) or torch.isinf(torch.tensor(fidelity)):
                print("⚠️ fidelity에서 NaN/Inf 감지, 0으로 대체")
                fidelity = 0.0
            if torch.isnan(torch.tensor(expressibility)) or torch.isinf(torch.tensor(expressibility)):
                print("⚠️ expressibility에서 NaN/Inf 감지, 0으로 대체")
                expressibility = 0.0
            
            # 수치 안정성 검증 (expressibility는 unbounded)
            result = {
                'entanglement': max(0.0, min(1.0, entanglement)),
                'fidelity': max(0.0, min(1.0, fidelity)),
                'expressibility': max(0.0, expressibility)  # expressibility는 [0, ∞) 범위
            }
            
            return result
    
    # Removed _prepare_circuit_data() method - dead code from legacy graph-based architecture
    # Current UnifiedPropertyPredictionTransformer handles CircuitSpec directly via _process_circuit_specs()
            

class AdaptiveGaussianRewardCalculator:
    """적응적 가우시안 리워드 계산기"""
    
    def __init__(self, property_weights: Optional[Dict[str, float]] = None,
                 property_sigmas: Optional[Dict[str, float]] = None):
        """
        Args:
            property_weights: 속성별 가중치
            property_sigmas: 속성별 기본 시그마 값
        """
        self.property_weights = property_weights or {
            'entanglement': 0.3,
            'expressibility': 0.4,
            'fidelity': 0.3
        }
        self.property_sigmas = property_sigmas or {
            'entanglement': 0.1,
            'expressibility': 0.1,
            'fidelity': 0.1
        }
        self.prediction_history = {}  # 예측 히스토리 저장
    
    def calculate_gaussian_reward(self, predicted_properties: Dict[str, float],
                                target_properties: Dict[str, float],
                                prediction_history: Optional[Dict[str, List[float]]] = None) -> float:
        """
        적응적 가우시안 리워드 계산
        
        Args:
            predicted_properties: 예측된 속성값들
            target_properties: 타겟 속성값들
            prediction_history: 예측 히스토리 (선택적)
            
        Returns:
            계산된 리워드
        """
        total_reward = 0.0
        
        for prop_name in self.property_weights:
            if prop_name in predicted_properties and prop_name in target_properties:
                pred_val = predicted_properties[prop_name]
                target_val = target_properties[prop_name]
                weight = self.property_weights[prop_name]
                
                # 거리 계산
                distance = abs(pred_val - target_val)
                
                # 적응적 sigma 계산 (선택적)
                if prediction_history and prop_name in prediction_history:
                    # 최근 예측 오차들의 표준편차를 sigma로 사용
                    recent_errors = prediction_history[prop_name][-10:]  # 최근 10개
                    adaptive_sigma = max(np.std(recent_errors), 0.01)  # 최소값 보장
                else:
                    # 기본 sigma 사용
                    adaptive_sigma = self.property_sigmas.get(prop_name, 0.1)
                
                # 가우시안 리워드 계산
                gaussian_reward = math.exp(-(distance**2) / (2 * adaptive_sigma**2))
                
                # 가중치 적용
                reward = weight * gaussian_reward
                total_reward += reward
        
        return total_reward
    
    def calculate_reward(self, predicted_properties: Dict[str, float],
                        target_properties: Dict[str, float]) -> float:
        """기본 리워드 계산 (항상 적응형 사용)"""
        return self.calculate_gaussian_reward(predicted_properties, target_properties, self.prediction_history)
    
    def update_prediction_history(self, prop_name: str, error: float):
        """예측 히스토리 업데이트"""
        if prop_name not in self.prediction_history:
            self.prediction_history[prop_name] = []
        self.prediction_history[prop_name].append(error)
        # 최대 50개까지만 유지
        if len(self.prediction_history[prop_name]) > 50:
            self.prediction_history[prop_name] = self.prediction_history[prop_name][-50:]
    
    def calculate_step_reward(self,
                            current_properties: Dict[str, float],
                            previous_properties: Dict[str, float],
                            target_properties: Dict[str, float]) -> float:
        """
        스텝별 리워드 계산 (개선도 기반, 가우시안 적용)
        
        Args:
            current_properties: 현재 스텝의 속성값들
            previous_properties: 이전 스텝의 속성값들
            target_properties: 타겟 속성값들
            
        Returns:
            계산된 스텝 리워드
        """
        # 현재와 이전 스텝의 가우시안 리워드 계산
        current_reward = self.calculate_gaussian_reward(current_properties, target_properties, self.prediction_history)
        previous_reward = self.calculate_gaussian_reward(previous_properties, target_properties, self.prediction_history)
        
        # 개선도 기반 리워드 (개선되면 양수, 악화되면 음수)
        improvement = current_reward - previous_reward
        
        # 베이스라인 리워드 추가 (0 수렴 방지)
        baseline_reward = 0.1
        step_reward = improvement + baseline_reward
        
        # 예측 히스토리 업데이트
        for prop_name in target_properties:
            if prop_name in current_properties:
                error = abs(current_properties[prop_name] - target_properties[prop_name])
                self.update_prediction_history(prop_name, error)
        
        return step_reward


class RTGCalculator:
    """모듈화된 RTG 계산기"""
    
    def __init__(self, 
                 property_predictor: PropertyPredictor,
                 reward_calculator: Optional[AdaptiveGaussianRewardCalculator] = None):
        """
        Args:
            property_predictor: 속성 예측기
            reward_calculator: 리워드 계산기
        """
        self.property_predictor = property_predictor
        self.reward_calculator = reward_calculator or AdaptiveGaussianRewardCalculator()
    
    def calculate_rtg_sequence(self,
                             circuit_specs: List,
                             target_properties: Dict[str, float],
                             gamma: float = 0.99) -> List[float]:
        """
        회로 시퀀스에 대한 RTG 계산
        
        Args:
            circuit_specs: 회로 스펙 리스트
            target_properties: 타겟 속성값들
            gamma: 할인 인자
            
        Returns:
            RTG 값들의 리스트
        """
        # 각 스텝별 속성 예측
        predicted_properties_sequence = []
        total_circuits = len(circuit_specs)
        for spec in circuit_specs:
            # 빈 회로(게이트 0개)는 기본값 사용 - NaN 방지
            if not hasattr(spec, 'gates') or len(spec.gates) == 0:
                props = {'entanglement': 0.0, 'fidelity': 0.0, 'expressibility': 0.0}
            else:
                # UnifiedPropertyPredictionTransformer는 forward 메서드 사용
                if hasattr(self.property_predictor, 'predict'):
                    props = self.property_predictor.predict(spec)
                else:
                    # forward 메서드 사용 (UnifiedPropertyPredictionTransformer)
                    with torch.no_grad():
                        output = self.property_predictor.forward(spec)
                        # 출력에서 속성값 추출
                        if isinstance(output, dict):
                            props = {
                                'entanglement': float(output.get('entanglement', 0.5)),
                                'fidelity': float(output.get('fidelity', 0.5)),
                                'expressibility': float(output.get('expressibility', 0.5))
                            }
                        else:
                            # 기본값 사용
                            props = {'entanglement': 0.5, 'fidelity': 0.5, 'expressibility': 0.5}
            
            predicted_properties_sequence.append(props)
        
        # 리워드 계산 (적응형 히스토리 업데이트 포함)
        rewards = []
        for i, props in enumerate(predicted_properties_sequence):
            # 예측 히스토리 업데이트 (모든 스텝에서)
            for prop_name in target_properties:
                if prop_name in props:
                    error = abs(props[prop_name] - target_properties[prop_name])
                    self.reward_calculator.update_prediction_history(prop_name, error)
            
            if i == 0:
                # 첫 번째 스텝은 절대 리워드 (적응형 사용)
                reward = self.reward_calculator.calculate_gaussian_reward(
                    props, target_properties, self.reward_calculator.prediction_history
                )
            else:
                # 이후 스텝은 개선도 기반 리워드 (적응형 사용)
                reward = self.reward_calculator.calculate_step_reward(
                    props, predicted_properties_sequence[i-1], target_properties
                )
            rewards.append(reward)
        
        # RTG 계산 (역순으로) - 마지막 스텝에서 시작하여 누적
        rtg_values = []
        rtg = 0.0
        
        for reward in reversed(rewards):
            rtg = reward + gamma * rtg
            rtg_values.append(rtg)
        
        # 순서 복원 (첫 번째 스텝이 가장 높은 RTG 값을 가짐)
        rtg_values.reverse()
        
        # RTG 값이 모두 0에 수렴하지 않도록 최소값 보장
        min_rtg = 0.1
        rtg_values = [max(rtg, min_rtg) for rtg in rtg_values]
        
        return rtg_values
    
    def calculate_episode_rtg(self,
                            episode_data: Dict[str, Any],
                            target_properties: Dict[str, float]) -> Dict[str, List[float]]:
        """
        에피소드 전체에 대한 RTG 계산
        
        Args:
            episode_data: 에피소드 데이터
            target_properties: 타겟 속성값들
            
        Returns:
            RTG 값들과 메타데이터
        """
        circuit_specs = episode_data.get('circuit_specs', [])
        
        if not circuit_specs:
            return {'rtg_values': [], 'rewards': [], 'properties': []}
        
        # RTG 계산
        rtg_values = self.calculate_rtg_sequence(circuit_specs, target_properties)
        
        # 추가 정보 수집
        properties_sequence = []
        for spec in circuit_specs:
            props = self.property_predictor.predict(spec)
            properties_sequence.append(props)
        
        # 리워드 시퀀스 계산 (적응형 히스토리 사용)
        rewards = []
        for i, props in enumerate(properties_sequence):
            # 예측 히스토리 업데이트
            for prop_name in target_properties:
                if prop_name in props:
                    error = abs(props[prop_name] - target_properties[prop_name])
                    self.reward_calculator.update_prediction_history(prop_name, error)
            
            if i == 0:
                reward = self.reward_calculator.calculate_gaussian_reward(
                    props, target_properties, self.reward_calculator.prediction_history
                )
            else:
                reward = self.reward_calculator.calculate_step_reward(
                    props, properties_sequence[i-1], target_properties
                )
            rewards.append(reward)
        
        return {
            'rtg_values': rtg_values,
            'rewards': rewards,
            'properties': properties_sequence,
            'target_properties': target_properties
        }


def create_rtg_calculator(checkpoint_path: Optional[str] = None,
                         property_weights: Optional[Dict[str, float]] = None,
                         device: str = "cpu",
                         loaded_model = None) -> RTGCalculator:
    """
    RTG 계산기를 생성하는 팩토리 함수
    
    Args:
        checkpoint_path: 모델 체크포인트 경로
        property_weights: 속성별 가중치
        device: 디바이스 설정
        loaded_model: 이미 로드된 모델 (있는 경우 사용)
        
    Returns:
        설정된 RTG 계산기
    """
    # Property 예측기 생성 - 이미 로드된 모델이 있으면 재사용
    if loaded_model is not None:
        print("\u2705 이미 로드된 property prediction 모델 재사용")
        property_predictor = ModelBasedPropertyPredictor(checkpoint_path, device, loaded_model)
    else:
        property_predictor = ModelBasedPropertyPredictor(checkpoint_path, device)
    
    # 리워드 계산기 생성
    reward_calculator = AdaptiveGaussianRewardCalculator(property_weights)
    
    # RTG 계산기 생성
    rtg_calculator = RTGCalculator(property_predictor, reward_calculator)
    
    return rtg_calculator


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 RTG Calculator 테스트 시작")
    
    rtg_calc = create_rtg_calculator()
    print("✅ RTG Calculator 생성 성공")
    
    # 더미 데이터로 테스트
    dummy_target = {
        'entanglement': 0.8,
        'expressibility': 0.7,
        'fidelity': 0.9
    }
    
    print(f"🎯 타겟 속성: {dummy_target}")
    print("✅ RTG Calculator 테스트 완료")
