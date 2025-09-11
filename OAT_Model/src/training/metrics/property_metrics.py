"""
Property Prediction Metrics Module

양자 회로 특성 예측 모델의 성능 평가를 위한 메트릭 계산 모듈
"""
import math
import torch
from typing import Dict, List, Any
from sklearn.metrics import confusion_matrix


class PropertyMetricsCalculator:
    """양자 회로 특성 예측 메트릭 계산기"""
    
    def __init__(self):
        self.properties = ['entanglement', 'fidelity', 'expressibility']
    
    def calculate_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """추가 메트릭 계산 - 개선된 에러 처리와 통계"""
        if not predictions or not targets:
            print("\n[WARNING] 메트릭 계산을 위한 데이터가 없습니다.")
            return {}
            
        try:
            metrics = {}
            
            # 분석할 프로퍼티 지정
            available_props = [p for p in self.properties 
                             if all(p in pred and p in target 
                                   for pred, target in zip(predictions, targets))]
            
            if not available_props:
                print("\n[WARNING] 메트릭 계산을 위한 프로퍼티가 없습니다.")
                return {}
                
            print(f"\n[INFO] 분석할 프로퍼티: {available_props}")
            
            # 분류 성능 메트릭 추가 (fidelity만)
            if 'fidelity' in available_props:
                confusion_metrics = self._calculate_confusion_matrix(predictions, targets, 'fidelity')
                metrics.update(confusion_metrics)
                
                if confusion_metrics:
                    print(f"  - fidelity 분류 성능:")
                    print(f"    └─ Accuracy: {confusion_metrics.get('fidelity_classification_accuracy', 0):.4f}")
                    print(f"    └─ Precision: {confusion_metrics.get('fidelity_classification_precision', 0):.4f}")
                    print(f"    └─ Recall: {confusion_metrics.get('fidelity_classification_recall', 0):.4f}")
                    print(f"    └─ F1: {confusion_metrics.get('fidelity_classification_f1', 0):.4f}")
            
            # 프로퍼티별 메트릭 계산
            for prop in available_props:
                prop_metrics = self._calculate_property_metrics(predictions, targets, prop)
                metrics.update(prop_metrics)
            
            return metrics
            
        except Exception as e:
            print(f"\n[ERROR] 메트릭 계산 오류: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _calculate_property_metrics(self, predictions: List[Dict], targets: List[Dict], prop: str) -> Dict[str, float]:
        """개별 프로퍼티에 대한 메트릭 계산"""
        try:
            # 예측/타겟값 모으기 (유효한 값만)
            pred_values = []
            target_values = []
            
            for pred, target in zip(predictions, targets):
                if prop in pred and prop in target:
                    # NaN/Inf 체크
                    p_vals = pred[prop]
                    t_vals = target[prop]
                    
                    valid_indices = ~(torch.isnan(p_vals) | torch.isinf(p_vals) | 
                                    torch.isnan(t_vals) | torch.isinf(t_vals))
                    if valid_indices.any():
                        pred_values.append(p_vals[valid_indices])
                        target_values.append(t_vals[valid_indices])
            
            if not pred_values:
                print(f"  - {prop}: 유효한 값이 없습니다")
                return {}
            
            # 유효한 값만 모아서 텐서로 변환
            pred_tensor = torch.cat(pred_values)
            target_tensor = torch.cat(target_values)
            
            # 값 범위 확인
            pred_min, pred_max = pred_tensor.min().item(), pred_tensor.max().item()
            target_min, target_max = target_tensor.min().item(), target_tensor.max().item()
            
            metrics = {}
            
            # MAE (Mean Absolute Error)
            mae = torch.mean(torch.abs(target_tensor - pred_tensor)).item()
            metrics[f'{prop}_mae'] = mae
            
            # MSE (Mean Squared Error)
            mse = torch.mean((target_tensor - pred_tensor) ** 2).item()
            metrics[f'{prop}_mse'] = mse
            
            # RMSE (Root Mean Squared Error)
            rmse = math.sqrt(mse)
            metrics[f'{prop}_rmse'] = rmse
            
            # R² score 계산
            ss_res = torch.sum((target_tensor - pred_tensor) ** 2)
            ss_tot = torch.sum((target_tensor - torch.mean(target_tensor)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            r2 = max(min(r2.item(), 1.0), -1.0)  # R2는 -∞ ~ 1 범위지만, 표시를 위해 제한
            metrics[f'{prop}_r2'] = r2
            
            # 평균 편향 (Mean Bias)
            mean_bias = torch.mean(pred_tensor - target_tensor).item()
            metrics[f'{prop}_bias'] = mean_bias
            
            # 상관계수 (Pearson correlation)
            if len(pred_tensor) > 1:  # 상관계수는 2개 이상의 샘플 필요
                pred_std = torch.std(pred_tensor)
                target_std = torch.std(target_tensor)
                if pred_std > 0 and target_std > 0:  # 0으로 나누는 것 방지
                    cov = torch.mean((pred_tensor - torch.mean(pred_tensor)) * 
                                   (target_tensor - torch.mean(target_tensor)))
                    corr = cov / (pred_std * target_std)
                    metrics[f'{prop}_corr'] = corr.item()
            
            print(f"  - {prop}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, "
                  f"bias={mean_bias:.4f}, range=({pred_min:.2f}-{pred_max:.2f})")
            
            # 피델리티에 대한 추가 디버깅
            if prop == 'fidelity':
                # 예측값과 타겟값의 분포 확인
                unique_preds = torch.unique(pred_tensor).shape[0]
                unique_targets = torch.unique(target_tensor).shape[0]
                print(f"    └─ Unique predictions: {unique_preds}, Unique targets: {unique_targets}")
                
                # 첫 10개 값 비교
                if len(pred_tensor) >= 10:
                    print(f"    └─ First 10 predictions: {pred_tensor[:10].tolist()}")
                    print(f"    └─ First 10 targets: {target_tensor[:10].tolist()}")
                
                # 피델리티 < 1.0인 샘플들만 필터링
                non_perfect_metrics = self._calculate_non_perfect_metrics(pred_tensor, target_tensor, prop)
                metrics.update(non_perfect_metrics)
            
            return metrics
            
        except Exception as e:
            print(f"❌ {prop} 메트릭 계산 실패: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"{prop} 메트릭 계산 중 치명적 오류: {e}") from e
    
    def _calculate_non_perfect_metrics(self, pred_tensor: torch.Tensor, target_tensor: torch.Tensor, prop: str) -> Dict[str, float]:
        """1이 아닌 피델리티 값들에 대한 추가 메트릭"""
        metrics = {}
        
        # 피델리티 < 1.0인 샘플들만 필터링
        non_perfect_mask = target_tensor < 0.999  # 부동소수점 오차 고려
        
        if non_perfect_mask.sum() > 0:
            non_perfect_pred = pred_tensor[non_perfect_mask]
            non_perfect_target = target_tensor[non_perfect_mask]
            
            # 1이 아닌 피델리티 값들에 대한 메트릭
            non_perfect_mae = torch.mean(torch.abs(non_perfect_target - non_perfect_pred)).item()
            non_perfect_mse = torch.mean((non_perfect_target - non_perfect_pred) ** 2).item()
            non_perfect_rmse = math.sqrt(non_perfect_mse)
            
            # R² score for non-perfect fidelity
            if len(non_perfect_target) > 1:
                ss_res_np = torch.sum((non_perfect_target - non_perfect_pred) ** 2)
                ss_tot_np = torch.sum((non_perfect_target - torch.mean(non_perfect_target)) ** 2)
                r2_np = 1 - (ss_res_np / (ss_tot_np + 1e-8))
                r2_np = max(min(r2_np.item(), 1.0), -1.0)
                metrics[f'{prop}_non_perfect_r2'] = r2_np
            
            # 평균 편향 for non-perfect
            non_perfect_bias = torch.mean(non_perfect_pred - non_perfect_target).item()
            
            # 상관계수 for non-perfect
            if len(non_perfect_pred) > 1:
                np_pred_std = torch.std(non_perfect_pred)
                np_target_std = torch.std(non_perfect_target)
                if np_pred_std > 0 and np_target_std > 0:
                    np_cov = torch.mean((non_perfect_pred - torch.mean(non_perfect_pred)) * 
                                       (non_perfect_target - torch.mean(non_perfect_target)))
                    np_corr = np_cov / (np_pred_std * np_target_std)
                    metrics[f'{prop}_non_perfect_corr'] = np_corr.item()
            
            # 메트릭 저장
            metrics[f'{prop}_non_perfect_mae'] = non_perfect_mae
            metrics[f'{prop}_non_perfect_mse'] = non_perfect_mse
            metrics[f'{prop}_non_perfect_rmse'] = non_perfect_rmse
            metrics[f'{prop}_non_perfect_bias'] = non_perfect_bias
            metrics[f'{prop}_non_perfect_count'] = non_perfect_mask.sum().item()
            
            # 범위 정보
            np_pred_min, np_pred_max = non_perfect_pred.min().item(), non_perfect_pred.max().item()
            np_target_min, np_target_max = non_perfect_target.min().item(), non_perfect_target.max().item()
            
            print(f"    └─ Non-perfect {prop} ({non_perfect_mask.sum().item()} samples): "
                  f"MAE={non_perfect_mae:.4f}, RMSE={non_perfect_rmse:.4f}, "
                  f"bias={non_perfect_bias:.4f}, range=({np_pred_min:.3f}-{np_pred_max:.3f})")
        else:
            print(f"    └─ All {prop} values are perfect (≥0.999)")
            metrics[f'{prop}_non_perfect_count'] = 0
        
        return metrics
    
    def _calculate_confusion_matrix(self, predictions: List[Dict], targets: List[Dict], prop: str) -> Dict[str, float]:
        """분류 성능을 위한 컨퓨전 매트릭스 계산"""
        try:
            # 예측값과 타겟값 수집
            pred_classifications = []
            target_classifications = []
            
            for pred, target in zip(predictions, targets):
                if f'{prop}_classifier' in pred and prop in target:
                    # 분류 예측 (임계값을 0.95로 조정)
                    pred_class = (pred[f'{prop}_classifier'] > 0.95).float()
                    # 타겟 분류 (>=0.999면 Full, <0.999면 Non-Full)
                    target_class = (target[prop] >= 0.999).float()
                    
                    pred_classifications.extend(pred_class.tolist())
                    target_classifications.extend(target_class.tolist())
            
            if not pred_classifications:
                return {}
            
            # 컨퓨전 매트릭스 계산
            cm = confusion_matrix(target_classifications, pred_classifications, labels=[0, 1])
            
            # 메트릭 계산
            tn, fp, fn, tp = cm.ravel()
            
            # 정확도, 정밀도, 재현율, F1 점수
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                f'{prop}_classification_accuracy': accuracy,
                f'{prop}_classification_precision': precision,
                f'{prop}_classification_recall': recall,
                f'{prop}_classification_f1': f1,
                f'{prop}_confusion_tn': float(tn),
                f'{prop}_confusion_fp': float(fp),
                f'{prop}_confusion_fn': float(fn),
                f'{prop}_confusion_tp': float(tp),
                f'{prop}_classification_total_samples': len(pred_classifications)
            }
            
        except Exception as e:
            print(f"[ERROR] {prop} 컨퓨전 매트릭스 계산 오류: {e}")
            return {}


class DebugLogger:
    """디버깅 및 로깅 유틸리티"""
    
    @staticmethod
    def debug_predictions_vs_targets(predictions: Dict[str, torch.Tensor], 
                                   targets: Dict[str, torch.Tensor], batch_idx: int):
        """예측값과 정답 레이블 비교 디버깅"""
        print(f"\n[DEBUG] [배치 {batch_idx}] 예측 vs 정답 디버깅:")
        
        # 첫 번째 샘플만 분석
        sample_idx = 0
        
        for property_name in ['entanglement', 'fidelity', 'expressibility']:
            if property_name in predictions and property_name in targets:
                pred_val = predictions[property_name][sample_idx].item()
                target_val = targets[property_name][sample_idx].item()
                diff = abs(pred_val - target_val)
                
                print(f"  [DATA] {property_name:15s}: 예측={pred_val:7.4f}, 정답={target_val:7.4f}, 차이={diff:7.4f}")
        
        # Combined 예측 (3차원 벡터)
        if 'combined' in predictions and 'combined' in targets:
            pred_combined = predictions['combined'][sample_idx]
            target_combined = targets['combined'][sample_idx]
            
            print(f"  [DATA] {'combined':15s}:")
            property_names = ['entanglement', 'fidelity', 'expressibility']
            for i, prop_name in enumerate(property_names):
                if i < len(pred_combined) and i < len(target_combined):
                    pred_val = pred_combined[i].item()
                    target_val = target_combined[i].item()
                    diff = abs(pred_val - target_val)
                    print(f"    - {prop_name:13s}: 예측={pred_val:7.4f}, 정답={target_val:7.4f}, 차이={diff:7.4f}")
        
        # 예측값 범위 체크
        print(f"  [RANGE] 예측값 범위 체크:")
        for property_name, pred_tensor in predictions.items():
            if torch.is_tensor(pred_tensor):
                min_val = pred_tensor.min().item()
                max_val = pred_tensor.max().item()
                mean_val = pred_tensor.mean().item()
                print(f"    - {property_name:13s}: min={min_val:7.4f}, max={max_val:7.4f}, mean={mean_val:7.4f}")
        
        # NaN/Inf 체크
        nan_found = False
        for property_name, pred_tensor in predictions.items():
            if torch.is_tensor(pred_tensor):
                if torch.isnan(pred_tensor).any() or torch.isinf(pred_tensor).any():
                    print(f"  [WARNING] {property_name}에서 NaN/Inf 감지!")
                    nan_found = True
        
        if not nan_found:
            print(f"  [OK] 모든 예측값이 정상 범위 내에 있습니다.")
        
        print()  # 빈 줄 추가
