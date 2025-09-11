"""
2024-2025 최신 연구 기반 완전 독립적 Property Prediction Head (Z-score 통일판)
- 모든 속성(얽힘도/피델리티/표현력) z-score 공간에서 직접 예측 (최종층 Linear)
- 학습 시: 손실/메트릭을 배치 z-score 기준으로 계산
- 추론 시: EMA 러닝 통계(mean/std)로 원 스케일 역변환 옵션 제공
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Blocks
# ----------------------------
class TaskSpecificNormalization(nn.Module):
    """LayerNorm + learnable scale/shift (task별 독립 파라미터)"""
    def __init__(self, dim: int, task_type: str):
        super().__init__()
        self.task_type = task_type
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=True)
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.layer_norm(x)
        return x * self.scale + self.shift


class IndependentTaskHead(nn.Module):
    """
    완전 독립적 태스크 헤드
    - Z-score 통일: 최종 출력 Linear(무제약) -> z-space 직접 예측
    - 내부 정규화는 LayerNorm 사용(BN 사용 금지)
    """
    def __init__(self, input_dim: int, task_type: str, dropout: float = 0.1):
        super().__init__()
        self.task_type = task_type
        self.network = self._build_unbounded_z_network(input_dim, dropout)

    def _build_unbounded_z_network(self, input_dim: int, dropout: float):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim),
            TaskSpecificNormalization(input_dim, self.task_type),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, 1),  # 최종 Linear: z-space 직접 예측
        )

    def forward(self, x):
        return self.network(x)


# ----------------------------
# Model
# ----------------------------
class OptimalPropertyHead(nn.Module):
    """Z-score 통일: 세 속성 모두 z-space에서 직접 예측"""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.gradient_hooks = []

        # 공통 피처 추출기
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        # 독립 예측 헤드(모두 z-space 출력)
        self.entanglement_head = IndependentTaskHead(d_model, 'entanglement', dropout)
        self.fidelity_head     = IndependentTaskHead(d_model, 'fidelity', dropout)
        self.expressibility_head = IndependentTaskHead(d_model, 'expressibility', dropout)

        # 속성별 EMA 러닝 통계 (추론 시 역변환용)
        # entanglement
        self.register_buffer('ent_mean', torch.tensor(0.0))
        self.register_buffer('ent_std',  torch.tensor(1.0))
        self.register_buffer('ent_count', torch.tensor(0))
        # fidelity
        self.register_buffer('fid_mean', torch.tensor(0.0))
        self.register_buffer('fid_std',  torch.tensor(1.0))
        self.register_buffer('fid_count', torch.tensor(0))
        # expressibility
        self.register_buffer('exp_mean', torch.tensor(0.0))
        self.register_buffer('exp_std',  torch.tensor(1.0))
        self.register_buffer('exp_count', torch.tensor(0))

        self.ema_momentum = 0.1  # 러닝 통계 EMA 모멘텀
        self.min_samples_for_stats = 10  # 통계 업데이트를 위한 최소 샘플 수

        self._init_weights()
        self._register_expressibility_hooks()  # 필요 시 디버깅용 훅

    # ----- Hooks (optional debugging) -----
    def _register_expressibility_hooks(self):
        # 훅 자체는 시그니처 유지 차원에서 남겨두되, 성능을 위해 기본적으로 등록 최소화
        for i, layer in enumerate(self.feature_extractor):
            if isinstance(layer, nn.Linear):
                hook = layer.register_full_backward_hook(
                    lambda module, gi, go, layer_idx=i:
                        self._layer_grad_hook(f"extractor_layer_{layer_idx}", module, gi, go)
                )
                self.gradient_hooks.append(hook)
        for i, layer in enumerate(self.expressibility_head.network):
            if isinstance(layer, nn.Linear):
                hook = layer.register_full_backward_hook(
                    lambda module, gi, go, layer_idx=i:
                        self._layer_grad_hook(f"head_layer_{layer_idx}", module, gi, go)
                )
                self.gradient_hooks.append(hook)

    def update_running_statistics(self, targets: Dict[str, torch.Tensor]):
        """훈련 중에만 실제 타겟 값들로 러닝 통계 업데이트"""
        if not self.training:
            return
            
        with torch.no_grad():
            for prop_name, values in targets.items():
                if values is None or values.numel() == 0:
                    continue
                    
                # NaN이나 inf 값 필터링
                valid_mask = torch.isfinite(values)
                if not valid_mask.any():
                    continue
                    
                valid_values = values[valid_mask].flatten()
                batch_mean = valid_values.mean()
                batch_std = valid_values.std() if valid_values.numel() > 1 else torch.tensor(1.0, device=values.device)
                
                # 속성별 통계 업데이트
                if prop_name == 'entanglement':
                    self._update_property_stats('ent', batch_mean, batch_std, valid_values.numel())
                elif prop_name == 'fidelity':
                    self._update_property_stats('fid', batch_mean, batch_std, valid_values.numel())
                elif prop_name == 'expressibility':
                    self._update_property_stats('exp', batch_mean, batch_std, valid_values.numel())
    
    def _update_property_stats(self, prefix: str, batch_mean: torch.Tensor, batch_std: torch.Tensor, batch_size: int):
        """개별 속성의 통계 업데이트 (EMA 방식)"""
        mean_buffer = getattr(self, f'{prefix}_mean')
        std_buffer = getattr(self, f'{prefix}_std')
        count_buffer = getattr(self, f'{prefix}_count')
        
        # 첫 번째 배치이거나 충분한 샘플이 없는 경우
        if count_buffer.item() < self.min_samples_for_stats:
            # 직접 업데이트
            mean_buffer.copy_(batch_mean)
            std_buffer.copy_(torch.max(batch_std, torch.tensor(1e-8, device=batch_std.device)))
            count_buffer.add_(batch_size)
        else:
            # EMA 업데이트
            mean_buffer.mul_(1 - self.ema_momentum).add_(batch_mean, alpha=self.ema_momentum)
            std_buffer.mul_(1 - self.ema_momentum).add_(batch_std, alpha=self.ema_momentum)
            # std가 너무 작아지지 않도록 최소값 보장
            std_buffer.clamp_(min=1e-8)
            count_buffer.add_(batch_size)

    def _layer_grad_hook(self, name, module, grad_input, grad_output):
        # 가벼운 체크(심한 로그는 사용자가 debug_mode로 감싸서 호출할 때만 권장)
        if grad_output and grad_output[0] is not None:
            gnorm = grad_output[0].norm().item()
            if gnorm < 1e-9 or gnorm > 1e3:
                print(f"[GRAD {name}] norm={gnorm:.3e} (vanish/explode risk)")

    def cleanup_hooks(self):
        for h in self.gradient_hooks:
            h.remove()
        self.gradient_hooks.clear()

    # ----- EMA stats -----
    @torch.no_grad()
    def _ema_update(self, mean_buf: torch.Tensor, std_buf: torch.Tensor, y: torch.Tensor, mom: float):
        bmean = y.mean()
        bvar  = y.var(unbiased=False)
        if torch.isfinite(bvar):
            mean_buf.mul_(1 - mom).add_(mom * bmean)
            new_var = (1 - mom) * (std_buf ** 2) + mom * bvar
            std_buf.copy_(torch.sqrt(torch.clamp(new_var, min=1e-6)))

    def update_entanglement_stats(self, entanglement_targets: torch.Tensor):
        if self.training:
            self._ema_update(self.ent_mean, self.ent_std, entanglement_targets, self.ema_momentum)

    def update_expressibility_stats(self, expressibility_targets: torch.Tensor):
        if self.training:
            self._ema_update(self.exp_mean, self.exp_std, expressibility_targets, self.ema_momentum)

    def _init_weights(self):
        def init_module(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=math.sqrt(2.0))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(init_module)

    # ----- Forward -----
    def forward(
        self,
        x: torch.Tensor,
        inference_mode: bool = False,
        debug_mode: bool = False,
        targets: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, d_model]
            inference_mode: True면 z->raw 역변환하여 반환
            debug_mode: 상세 로그 (기본 False)
            targets: 학습 시 러닝 통계 업데이트용 (선택)
        Returns:
            dict: {'entanglement': z_pred, 'fidelity': z_pred, 'expressibility': z_pred} (기본 z-space)
                  inference_mode=True면 raw-space로 역변환하여 반환
        """
        if debug_mode:
            with torch.no_grad():
                print(f"[DEBUG] x: shape={x.shape} min={x.min():.3f} max={x.max():.3f} mean={x.mean():.3f}")

        features = self.feature_extractor(x)

        ent_z = self.entanglement_head(features)
        fid_z = self.fidelity_head(features)
        exp_z = self.expressibility_head(features)

        preds = {
            'entanglement': ent_z,
            'fidelity':     fid_z,
            'expressibility': exp_z
        }

        # 학습 중 러닝 통계 업데이트 (새로운 통합 방식)
        if self.training and targets is not None:
            self.update_running_statistics(targets)

        # 추론 모드: z -> raw 역변환
        if inference_mode:
            with torch.no_grad():
                preds = {
                    'entanglement': preds['entanglement'] * self.ent_std + self.ent_mean,
                    'fidelity':     preds['fidelity']     * self.fid_std + self.fid_mean,
                    'expressibility': preds['expressibility'] * self.exp_std + self.exp_mean,
                }

        return preds

    def _validate_statistics(self) -> bool:
        """저장된 통계가 유효한지 검증"""
        # 최소 샘플 수 확인
        min_samples_ok = (
            self.ent_count.item() >= self.min_samples_for_stats and
            self.fid_count.item() >= self.min_samples_for_stats and
            self.exp_count.item() >= self.min_samples_for_stats
        )
        
        # 표준편차가 유효한지 확인 (너무 작거나 정확히 1.0이면 의심스러움)
        std_valid = (
            self.ent_std.item() > 1e-6 and self.ent_std.item() != 1.0 and
            self.fid_std.item() > 1e-6 and self.fid_std.item() != 1.0 and
            self.exp_std.item() > 1e-6 and self.exp_std.item() != 1.0
        )
        
        # 평균이 정확히 0.0이 아닌지 확인 (적어도 하나는 0이 아니어야 함)
        mean_valid = not (
            abs(self.ent_mean.item()) < 1e-8 and
            abs(self.fid_mean.item()) < 1e-8 and
            abs(self.exp_mean.item()) < 1e-8
        )
        
        return min_samples_ok and std_valid and mean_valid


# ----------------------------
# Loss
# ----------------------------
class OptimalPropertyLoss(nn.Module):
    """
    Z-score 통일: 손실은 배치 z-score 기준으로 계산
    - 시그니처 유지: exp_mean/exp_std 인자는 무시(하위호환)
    """
    def __init__(self, property_weights: Optional[Dict[str, float]] = None, huber_delta: float = 2.0):
        super().__init__()
        self.property_weights = property_weights or {
            'entanglement': 1.0,
            'fidelity':     1.0,
            'expressibility': 1.0,
        }
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.mae   = nn.L1Loss()
        self.mse   = nn.MSELoss()

    @staticmethod
    def _to_batch_z(y: torch.Tensor):
        mu = y.mean()
        sd = torch.sqrt(torch.clamp(y.var(unbiased=False), min=1e-6))
        return (y - mu) / sd, mu, sd

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        exp_mean: torch.Tensor = None,   # <- 유지(미사용)
        exp_std: torch.Tensor = None,    # <- 유지(미사용)
        debug_mode: bool = False
    ) -> tuple:
        """
        Returns:
            total_loss (Tensor), individual_losses (dict)
        """
        losses = {}
        total_loss = torch.zeros((), device=list(predictions.values())[0].device)

        for prop in ['entanglement', 'fidelity', 'expressibility']:
            if prop not in predictions or prop not in targets:
                continue

            y   = targets[prop].view_as(predictions[prop])  # [B,1] 정렬
            z_y, mu, sd = self._to_batch_z(y)
            z_pred = predictions[prop]

            # 메인 손실: Huber(z_pred, z_y)
            main = self.huber(z_pred, z_y)
            aux_mae = self.mae(z_pred, z_y)
            aux_mse = self.mse(z_pred, z_y)

            losses[f'{prop}_huber'] = main
            losses[f'{prop}_mae']   = aux_mae
            losses[f'{prop}_mse']   = aux_mse

            total_loss = total_loss + self.property_weights[prop] * main

            if debug_mode:
                with torch.no_grad():
                    print(f"[LOSS-{prop}] z_y mean={z_y.mean():.3f} std={z_y.std():.3f} | "
                          f"z_pred mean={z_pred.mean():.3f} std={z_pred.std():.3f} | huber={main.item():.4f}")

        return total_loss, losses


# ----------------------------
# Metrics
# ----------------------------
class PropertyMetrics:
    """메트릭: 배치 z-score 기준으로 계산 (시그니처 유지)"""
    @staticmethod
    def compute_metrics(
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        exp_mean: torch.Tensor = None,   # <- 유지(미사용)
        exp_std: torch.Tensor = None     # <- 유지(미사용)
    ) -> Dict[str, float]:
        metrics = {}

        def r2_score(y_true, y_pred):
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
            return (1 - ss_res / torch.clamp(ss_tot, min=1e-12)).item()

        for prop in ['entanglement', 'fidelity', 'expressibility']:
            if prop not in predictions or prop not in targets:
                continue

            z_pred = predictions[prop].detach().cpu()
            y = targets[prop].view_as(z_pred).detach().cpu()

            # 배치 z-score로 타깃 정규화
            mu = y.mean()
            sd = torch.sqrt(torch.clamp(y.var(unbiased=False), min=1e-6))
            z_y = (y - mu) / sd

            mae  = torch.mean(torch.abs(z_pred - z_y)).item()
            rmse = torch.sqrt(torch.mean((z_pred - z_y) ** 2)).item()
            r2   = r2_score(z_y, z_pred)

            metrics[f'{prop}_r2']   = r2
            metrics[f'{prop}_mae']  = mae
            metrics[f'{prop}_rmse'] = rmse

            # z-space 범위 참고(±3)
            within = torch.logical_and(z_pred >= -3, z_pred <= 3).float().mean().item()
            metrics[f'{prop}_zrange_frac'] = within

        return metrics
