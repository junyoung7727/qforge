"""
SOTA Property Prediction Transformer Training Pipeline
IntegratedPropertyPredictionTransformer 전용 모듈화된 트레이너
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import numpy as np
from tqdm import tqdm
import wandb
import gc

# Import SOTA model only
from models.integrated_property_prediction_transformer import IntegratedPropertyPredictionTransformer, IntegratedPropertyPredictionConfig

# Import circuit interface
sys.path.append(str(Path(__file__).parent.parent.parent / "quantumcommon"))


class PropertyPredictionTrainer:
    """SOTA Property Prediction Transformer 전용 학습기"""
    
    def __init__(
        self,
        model: IntegratedPropertyPredictionTransformer,
        config: IntegratedPropertyPredictionConfig,
        device: str = "auto",
        use_wandb: bool = True
    ):
        # 디바이스 설정
        self.device = self._get_device(device)
        
        # 모델 및 설정 (SOTA 모델만 지원)
        self.model = model.to(self.device)
        self.config = config
        
        # WandB 설정
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project="quantum-property-prediction",
                name=f"training_{int(time.time())}",
                config=config.__dict__ if hasattr(config, '__dict__') else {}
            )
        
        # 학습 컴포넌트 초기화
        self._setup_training_components()
        
        # 학습 상태 초기화
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        
        # 메모리 최적화 설정
        if self.device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        
        print(f"✅ SOTA Property Prediction Trainer 초기화 완료")
        print(f"   - 모델 타입: SOTA 통합 모델")
        print(f"   - 디바이스: {self.device}")
        print(f"   - 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _get_device(self, device: str) -> torch.device:
        """디바이스 설정"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _setup_training_components(self):
        """옵티마이저, 손실함수, 스케줄러 설정"""
        # 옵티마이저 설정
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate if hasattr(self.config, 'learning_rate') else 1e-4,
            weight_decay=self.config.weight_decay if hasattr(self.config, 'weight_decay') else 1e-5
        )
        
        # 워밍업 스케줄러 설정
        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        
        warmup_steps = getattr(self.config, 'warmup_steps', 100)
        total_epochs = getattr(self.config, 'num_epochs', 100)
        
        # 워밍업 스케줄러 (처음 100스텝 동안 선형 증가)
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # 메인 스케줄러 (코사인 어닐링 with restarts)
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        
        scheduler_type = getattr(self.config, 'scheduler_type', 'cosine')
        min_lr = getattr(self.config, 'min_learning_rate', 1e-6)
        
        if scheduler_type == "cosine_with_restarts":
            main_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=20,  # 첫 번째 재시작 주기
                T_mult=2,  # 주기 배수
                eta_min=min_lr  # 최소 학습률
            )
        else:
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs - warmup_steps,
                eta_min=min_lr
            )
        
        # 순차 스케줄러로 결합
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
        
        # 학습 상태
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        
    
    def _prepare_batch_data(self, batch):
        """배치 데이터 준비 - 유니파이드 러너에서 이동"""
        circuit_specs = batch.get('circuit_specs', [])
        
        if not circuit_specs:
            return {
                'circuit_data': None,
                'target_properties': None
            }
        
        # 타겟 속성 추출
        targets_dict = batch.get('targets')
        target_properties = None
        
        if targets_dict:
            batch_size = len(circuit_specs)
            target_properties = torch.zeros(batch_size, 3, device=self.device)
            
            # 타겟 값 추출 및 텐서 변환
            entanglement_vals = targets_dict.get('entanglement', torch.zeros(batch_size))
            expressibility_vals = targets_dict.get('expressibility', torch.zeros(batch_size))
            fidelity_vals = targets_dict.get('fidelity', torch.ones(batch_size))
            
            # 적절한 크기로 변환
            if entanglement_vals.dim() == 0:
                entanglement_vals = entanglement_vals.unsqueeze(0).expand(batch_size)
            if expressibility_vals.dim() == 0:
                expressibility_vals = expressibility_vals.unsqueeze(0).expand(batch_size)
            if fidelity_vals.dim() == 0:
                fidelity_vals = fidelity_vals.unsqueeze(0).expand(batch_size)
            
            target_properties[:, 0] = entanglement_vals.to(self.device)
            target_properties[:, 1] = expressibility_vals.to(self.device)
            target_properties[:, 2] = fidelity_vals.to(self.device)
        
        return {
            'circuit_specs': circuit_specs,
            'target_properties': target_properties
        }
    
    def _compute_loss(self, predictions, targets):
        """SOTA 모델 손실 계산"""
        # SOTA 모델의 내장 손실 함수 사용
        losses = self.model.compute_loss(predictions, targets)
        
        # 개별 손실 및 동적 가중치 로깅
        if hasattr(self, 'use_wandb') and self.use_wandb:
            import wandb
            log_dict = {}
            for loss_name, loss_value in losses.items():
                if loss_name != 'total' and isinstance(loss_value, torch.Tensor):
                    log_dict[f'train/{loss_name}_loss'] = loss_value.item()
            
            # 동적 가중치 로깅
            if hasattr(self.model, 'loss_function') and hasattr(self.model.loss_function, 'log_vars'):
                log_vars = self.model.loss_function.log_vars.data
                for i, prop in enumerate(['entanglement', 'fidelity', 'expressibility']):
                    precision = torch.exp(-log_vars[i])
                    log_dict[f'train/dynamic_weight_{prop}'] = precision.item()
                    log_dict[f'train/uncertainty_{prop}'] = torch.exp(log_vars[i]).item()
            
            if log_dict:
                wandb.log(log_dict, step=self.current_step)
        
        # 콘솔 로깅
        individual_losses = []
        for loss_name, loss_value in losses.items():
            if loss_name != 'total' and isinstance(loss_value, torch.Tensor):
                individual_losses.append(f"{loss_name}: {loss_value.item():.4f}")
        
        # 동적 가중치 및 스케일 표시
        if hasattr(self.model, 'loss_function') and hasattr(self.model.loss_function, 'log_vars'):
            log_vars = self.model.loss_function.log_vars.data
            loss_scales = self.model.loss_function.loss_scales
            
            weights_info = []
            scales_info = []
            for i, prop in enumerate(['ent', 'fid', 'exp']):
                weight = torch.exp(-log_vars[i]).item()
                scale = loss_scales[['entanglement', 'fidelity', 'expressibility'][i]]
                weights_info.append(f"{prop}:{weight:.3f}")
                scales_info.append(f"{prop}:{scale:.3f}")
            
            individual_losses.append(f"weights[{','.join(weights_info)}]")
            individual_losses.append(f"scales[{','.join(scales_info)}]")
        
        if individual_losses and self.current_step % 50 == 0:  # 50스텝마다 로깅
            print(f"📊 Step {self.current_step} - Individual losses: {', '.join(individual_losses)}")
        
        return losses['total']
    
    def _forward_step(self, prepared_batch):
        """SOTA 모델 forward pass"""
        circuit_specs = prepared_batch['circuit_specs']
        target_properties = prepared_batch['target_properties']
        
        if circuit_specs is None or target_properties is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 타겟을 딕셔너리 형태로 변환
        targets_dict = {
            'entanglement': target_properties[:, 0],
            'expressibility': target_properties[:, 1],
            'fidelity': target_properties[:, 2]
        }
        
        # Forward pass (targets 전달하여 통계 업데이트)
        outputs = self.model(circuit_specs, targets=targets_dict)
        
        # Loss 계산
        loss = self._compute_loss(outputs, targets_dict)
        return loss
    
    def train_epoch(self, train_loader):
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        # 메모리 정리
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 그래디언트 초기화
                self.optimizer.zero_grad()
                
                # 배치 데이터 준비
                prepared_batch = self._prepare_batch_data(batch)
                
                # Forward pass
                loss = self._forward_step(prepared_batch)
                
                # NaN/Inf 검사
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ Invalid loss at batch {batch_idx}, skipping...")
                    continue
                
                # Backward pass (모든 모델에 대해 동일)
                loss.backward()
                
                # 그래디언트 클리핑
                if hasattr(self.config, 'gradient_clipping') and self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                
                # 옵티마이저 스텝
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                loss_value = loss.item()
                del loss
                
                # 손실 누적
                total_loss += loss_value
                self.current_step += 1
                
                # Progress bar 업데이트
                pbar.set_postfix({'loss': f'{loss_value:.4f}'})
                
                # 메모리 정리
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ CUDA memory error at batch {batch_idx}, clearing cache...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def _validation_step(self, prepared_batch):
        """검증용 스텝 - 향상된 디버깅 포함"""
        circuit_specs = prepared_batch['circuit_specs']
        target_properties = prepared_batch['target_properties']
        
        if circuit_specs is None or target_properties is None:
            return 0.0
        
        # Forward pass
        outputs = self.model(circuit_specs)
        
        # 타겟을 딕셔너리 형태로 변환
        targets_dict = {
            'entanglement': target_properties[:, 0],
            'expressibility': target_properties[:, 1],
            'fidelity': target_properties[:, 2]
        }
        
        # Enhanced debugging for validation data
        if hasattr(self, '_val_step_counter'):
            self._val_step_counter += 1
        else:
            self._val_step_counter = 1
            
        # Debug validation data every 5 steps
        if self._val_step_counter % 5 == 0:
            print(f"\n[VALIDATION STEP {self._val_step_counter}] Data Analysis:")
            if isinstance(circuit_specs, list):
                print(f"  Circuit specs: list with {len(circuit_specs)} items")
            else:
                print(f"  Circuit specs shape: {circuit_specs.shape}")
            print(f"  Target properties shape: {target_properties.shape}")
            
            # Check target expressibility values
            exp_targets = targets_dict['expressibility']
            print(f"  Expressibility targets: min={exp_targets.min().item():.8f}, max={exp_targets.max().item():.8f}")
            print(f"  Expressibility targets: mean={exp_targets.mean().item():.8f}, std={exp_targets.std().item():.8f}")
            
            # Check model outputs
            if 'expressibility' in outputs:
                exp_outputs = outputs['expressibility']
                print(f"  Expressibility outputs: min={exp_outputs.min().item():.8f}, max={exp_outputs.max().item():.8f}")
                print(f"  Expressibility outputs: mean={exp_outputs.mean().item():.8f}, std={exp_outputs.std().item():.8f}")
        
        # 검증에서는 개별 손실도 함께 반환
        losses = self.model.compute_loss(outputs, targets_dict)
        
        # Enhanced loss logging with detailed analysis
        if hasattr(self, 'use_wandb') and self.use_wandb:
            import wandb
            log_dict = {}
            
            # Debug loss computation
            if self._val_step_counter % 5 == 0:
                print(f"\n[VALIDATION LOSS] Computed losses:")
            
            for loss_name, loss_value in losses.items():
                if loss_name != 'total' and isinstance(loss_value, torch.Tensor):
                    loss_val = loss_value.item()
                    log_dict[f'val/{loss_name}_loss'] = loss_val
                    
                    # Debug specific loss values
                    if self._val_step_counter % 5 == 0:
                        print(f"  {loss_name}: {loss_val:.8f}")
                        
                        # Special attention to expressibility
                        if loss_name == 'expressibility':
                            if hasattr(self, '_prev_exp_loss'):
                                if abs(loss_val - self._prev_exp_loss) < 1e-10:
                                    print(f"  WARNING: Expressibility loss unchanged from previous: {self._prev_exp_loss:.8f}")
                                else:
                                    print(f"  Expressibility loss changed from: {self._prev_exp_loss:.8f} to {loss_val:.8f}")
                            self._prev_exp_loss = loss_val
            
            if log_dict:
                wandb.log(log_dict, step=self.current_step)
                
                # Debug wandb logging
                if self._val_step_counter % 5 == 0:
                    print(f"  Logged to wandb: {list(log_dict.keys())}")
        
        return losses['total'].item() if hasattr(losses['total'], 'item') else float(losses['total'])

    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        # 개별 손실 집계를 위한 딕셔너리
        accumulated_losses = {}
        valid_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch_idx, batch in enumerate(pbar):
                try:
                    # 배치 데이터 준비
                    prepared_batch = self._prepare_batch_data(batch)
                    
                    # Forward pass with detailed losses
                    circuit_specs = prepared_batch['circuit_specs']
                    target_properties = prepared_batch['target_properties']
                    
                    if circuit_specs is None or target_properties is None:
                        continue
                    
                    # Forward pass
                    outputs = self.model(circuit_specs)
                    
                    # 타겟을 딕셔너리 형태로 변환
                    targets_dict = {
                        'entanglement': target_properties[:, 0],
                        'expressibility': target_properties[:, 1],
                        'fidelity': target_properties[:, 2]
                    }
                    
                    # 손실 계산
                    losses = self.model.compute_loss(outputs, targets_dict)
                    total_loss_value = losses['total'].item() if hasattr(losses['total'], 'item') else float(losses['total'])
                    
                    # NaN/Inf 검사
                    if torch.isnan(torch.tensor(total_loss_value)) or torch.isinf(torch.tensor(total_loss_value)):
                        print(f"⚠️ Warning: Invalid loss detected in validation batch {batch_idx}: {total_loss_value}")
                        continue
                    
                    total_loss += total_loss_value
                    valid_batches += 1
                    
                    # 개별 손실 집계
                    for loss_name, loss_value in losses.items():
                        if loss_name != 'total' and isinstance(loss_value, torch.Tensor):
                            loss_val = loss_value.item()
                            if loss_name not in accumulated_losses:
                                accumulated_losses[loss_name] = 0.0
                            accumulated_losses[loss_name] += loss_val
                    
                    pbar.set_postfix({'val_loss': f'{total_loss_value:.4f}'})
                    
                except Exception as e:
                    print(f"❌ Error in validation batch {batch_idx}: {e}")
                    continue
        
        # 평균 계산
        avg_loss = total_loss / max(valid_batches, 1)
        
        # 개별 손실 평균 계산 및 wandb 로깅
        if hasattr(self, 'use_wandb') and self.use_wandb and accumulated_losses:
            import wandb
            val_log_dict = {}
            
            for loss_name, total_loss_val in accumulated_losses.items():
                avg_loss_val = total_loss_val / max(valid_batches, 1)
                val_log_dict[f'val/{loss_name}_loss_epoch'] = avg_loss_val
            
            val_log_dict['val/total_loss_epoch'] = avg_loss
            val_log_dict['epoch'] = self.current_epoch
            
            wandb.log(val_log_dict, step=self.current_step)
            print(f"📊 Validation losses logged to wandb: {list(val_log_dict.keys())}")
        
        return avg_loss
    
    def train(self, train_loader, val_loader=None, num_epochs=100):
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 학습
            train_loss = self.train_epoch(train_loader)
            
            # 검증
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                
                # WandB 로깅
                if self.use_wandb:
                    wandb.log({
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'epoch': epoch,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
                
                # 최고 모델 저장
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    print(f"✅ 새로운 최고 모델 (val_loss: {val_loss:.4f})")
                    
                    # Save best model checkpoint (상대 경로)
                    checkpoint_path = Path("./checkpoints/best_model.pt")
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    self.save_checkpoint(str(checkpoint_path))
                    print(f"✓ 최고 모델 체크포인트가 저장되었습니다: {checkpoint_path}")
                    print(f"✓ 절대 경로: {checkpoint_path.resolve()}")
                    
                    if self.use_wandb:
                        wandb.log({"best_model_saved": True, "best_val_loss": val_loss})
            else:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
                
                if self.use_wandb:
                    wandb.log({
                        'train_loss': train_loss,
                        'epoch': epoch,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
            
            # 스케줄러 스텝
            if self.scheduler:
                self.scheduler.step()
        
        print("🎉 학습 완료!")
        
        if self.use_wandb:
            wandb.finish()
        
        return {
            'best_val_loss': self.best_val_loss,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss
        }
    
    def save_checkpoint(self, filepath):
        """체크포인트 저장 - 모델 설정 포함"""
        # 기본 체크포인트 정보
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'best_val_loss': self.best_val_loss,
        }
        
        # 모델 구성 정보 저장
        if hasattr(self.model, 'get_model_info'):
            # IntegratedPropertyPredictionTransformer의 get_model_info 메서드 사용
            model_info = self.model.get_model_info()
            checkpoint['model_info'] = model_info
        
        # 모델 구성 클래스 타입 저장
        checkpoint['model_class'] = self.model.__class__.__name__
        
        # 설정 저장 (dataclass 또는 dict 형태)
        if hasattr(self.config, '__dict__'):
            checkpoint['config'] = self.config.__dict__
        elif hasattr(self.config, '_asdict'):
            checkpoint['config'] = self.config._asdict()
        else:
            checkpoint['config'] = self.config if isinstance(self.config, dict) else {}
            
        # 모델의 설정 객체 종류 저장
        if hasattr(self.config, '__class__'):
            checkpoint['config_class'] = self.config.__class__.__name__
        
        torch.save(checkpoint, filepath)
        print(f"✅ 설정 포함 체크포인트 저장: {filepath}")
        print(f"   모델 클래스: {checkpoint['model_class']}")
        if 'config_class' in checkpoint:
            print(f"   설정 클래스: {checkpoint['config_class']}")
        else:
            print(f"   설정: {type(checkpoint['config'])}")
            
        # 버퍼 통계 저장 로깅 (expressibility 통계 확인)
        if hasattr(self.model, 'prediction_head') and hasattr(self.model.prediction_head, 'exp_mean'):
            exp_mean = getattr(self.model.prediction_head, 'exp_mean', None)
            exp_std = getattr(self.model.prediction_head, 'exp_std', None)
            if exp_mean is not None and exp_std is not None:
                print(f"   Expressibility 통계 - mean: {exp_mean.item():.6f}, std: {exp_std.item():.6f}")
        
    
    def load_checkpoint(self, filepath):
        """체크포인트 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.current_step = checkpoint.get('current_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"체크포인트 로드: {filepath}")
        print(f"에폭 {self.current_epoch}부터 재시작")


def create_trainer(model, config, device="auto", use_wandb=True):
    """트레이너 팩토리 함수"""
    return PropertyPredictionTrainer(
        model=model,
        config=config,
        device=device,
        use_wandb=use_wandb
    )