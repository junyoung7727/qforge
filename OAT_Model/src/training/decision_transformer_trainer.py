"""
Decision Transformer 전용 트레이너
완전히 새로운 깔끔한 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Tuple
import time
import json
from torch import optim
from pathlib import Path
import wandb

from models.decision_transformer import DecisionTransformer
from rtg.core.rtg_calculator import RTGCalculator
from utils.device_manager import get_device_manager, cleanup_memory, get_memory_info


class DecisionTransformerTrainer:
    """Decision Transformer 전용 트레이너"""
    
    def __init__(
        self,
        model: DecisionTransformer,
        config: Dict,
        device: str = "cuda",
        use_wandb: bool = True
    ):
        # Device manager 초기화
        self.device_manager = get_device_manager(device)
        self.device = self.device_manager.device
        self.model = self.device_manager.move_model_to_device(model)
        self.config = config
        self.use_wandb = use_wandb
        
        # 메모리 정리 설정
        self.cleanup_interval = config.get('memory_cleanup_interval', 50)  # 50 배치마다
        self.memory_threshold_gb = config.get('memory_threshold_gb', 18.0)  # 8GB 임계값
        self.last_cleanup_batch = 0
        
        # 성능 최적화 설정 (그래디언트 누적 비활성화로 backward 오류 방지)
        self.accumulate_grad_batches = 1  # 그래디언트 누적 비활성화
        self.mixed_precision = config.get('mixed_precision', False)  # 혼합 정밀도 비활성화
        
        # Mixed precision scaler
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # 옵티마이저 설정
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01),
            betas=(0.9, 0.95)
        )
        
        # 스케줄러 설정
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('max_epochs', 100),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Huber Loss for property prediction
        self.huber_loss = nn.HuberLoss(delta=1.0)
        
        # 훈련 상태
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # WandB 초기화
        if use_wandb:
            wandb.init(
                project="decision-transformer-quantum",
                name=f"dt_clean_{int(time.time())}",
                config=config,
                tags=["decision_transformer", "quantum_circuit", "gate_prediction"]
            )
            # 모델 아키텍처 로깅
            wandb.watch(self.model, log="all", log_freq=100)
        
        print(f"✅ Decision Transformer Trainer 초기화 완료")
        print(f"   - 디바이스: {self.device}")
        print(f"   - 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """한 에포크 훈련 - 완전 리팩토링"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 단순한 훈련 스텝
            loss_value = self._train_step(batch)
            total_loss += loss_value
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': f'{loss_value:.4f}', 
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # 로깅
            if self.use_wandb and batch_idx % 10 == 0:
                self._log_metrics(loss_value)
            
            self.global_step += 1
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                try:
                    input_sequence = batch['input_sequence'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    action_prediction_mask = batch['action_prediction_mask'].to(self.device)
                    target_properties = batch['target_properties'].to(self.device)
                    
                    # Forward pass
                    predictions = self.model(
                        input_sequence=input_sequence,
                        attention_mask=attention_mask,
                        action_prediction_mask=action_prediction_mask
                    )
                    
                    # Loss 계산
                    loss = self._compute_loss(predictions, batch, action_prediction_mask)
                    total_loss += loss.detach().item()
                    
                except Exception as e:
                    print(f"❌ Validation error in batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """단일 훈련 스텝 - 깔끔한 구현"""
        # 그래디언트 초기화
        self.optimizer.zero_grad()
        
        # 입력 준비
        inputs = self._prepare_inputs(batch)
        
        # Forward pass
        predictions = self.model(**inputs)
        
        # Loss 계산
        loss = self._compute_loss(predictions, batch, inputs['action_prediction_mask'])
        
        # Backward pass - detach loss to prevent graph reuse
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        return loss.detach().item()
    
    def _compute_gate_loss(self, gate_pred: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """게이트 예측 손실"""
        actions = actions.to(self.device)
        valid_positions = mask.bool()
        
        if not valid_positions.any():
            return None
            
        # 유효한 위치의 예측값
        gate_logits = gate_pred[valid_positions]
        
        # 타겟 준비 - SAR 매핑, 스칼라 값만 추출
        targets = []
        batch_size, sar_seq_len = mask.shape
        _, action_seq_len = actions.shape[:2]
        
        for b in range(batch_size):
            for sar_idx in range(sar_seq_len):
                if mask[b, sar_idx]:
                    action_idx = sar_idx // 3
                    if action_idx < action_seq_len:
                        # 게이트 ID는 스칼라 값이어야 함
                        if len(actions.shape) > 2:
                            targets.append(actions[b, action_idx, 0])  # 첫 번째 요소 (게이트 ID)
                        else:
                            targets.append(actions[b, action_idx])
        
        if len(targets) == 0:
            return None
            
        targets_tensor = torch.stack(targets).to(self.device).long()
        
        # Debug: Check target values range
        max_target = targets_tensor.max().item()
        min_target = targets_tensor.min().item()
        vocab_size = gate_logits.shape[-1]
        
        if max_target >= vocab_size or min_target < 0:
            print(f"⚠️ Target out of bounds: min={min_target}, max={max_target}, vocab_size={vocab_size}")
            # Clamp targets to valid range
            targets_tensor = torch.clamp(targets_tensor, 0, vocab_size - 1)
        
        return F.cross_entropy(gate_logits, targets_tensor)
    
    def _compute_position_loss(self, pos_pred: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """위치 예측 손실 - actions에서 qubit 위치 추출"""
        actions = actions.to(self.device)
        valid_positions = mask.bool()
        
        if not valid_positions.any():
            return None
            
        # 유효한 위치의 예측값 (처음 2차원만 사용)
        pos_logits = pos_pred[valid_positions][:, :2]
        
        # 타겟 준비 - actions에서 qubit 위치 추출 [qubit1, qubit2]
        targets = []
        batch_size, sar_seq_len = mask.shape
        _, action_seq_len = actions.shape[:2]
        
        for b in range(batch_size):
            for sar_idx in range(sar_seq_len):
                if mask[b, sar_idx]:
                    action_idx = sar_idx // 3
                    if action_idx < action_seq_len:
                        if len(actions.shape) > 2:
                            # actions[gate_id, qubit1, qubit2, param]에서 qubit1, qubit2 추출
                            qubit_positions = actions[b, action_idx, 1:3]  # [qubit1, qubit2]
                            targets.append(qubit_positions)
                        else:
                            # 2D인 경우 전체 사용 (fallback)
                            targets.append(actions[b, action_idx])
        
        if len(targets) == 0:
            return None
            
        targets_tensor = torch.stack(targets).to(self.device)
        return F.mse_loss(pos_logits, targets_tensor.float())
    
    def _compute_parameter_loss(self, param_pred: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """파라미터 예측 손실 - actions에서 파라미터 추출"""
        actions = actions.to(self.device)
        valid_positions = mask.bool()
        
        if not valid_positions.any():
            return None
            
        # 유효한 위치의 예측값
        param_logits = param_pred[valid_positions].squeeze(-1)
        
        # 타겟 준비 - actions에서 파라미터 추출
        targets = []
        batch_size, sar_seq_len = mask.shape
        _, action_seq_len = actions.shape[:2]
        
        for b in range(batch_size):
            for sar_idx in range(sar_seq_len):
                if mask[b, sar_idx]:
                    action_idx = sar_idx // 3
                    if action_idx < action_seq_len:
                        if len(actions.shape) > 2:
                            # actions[gate_id, qubit1, qubit2, param]에서 param 추출
                            param_value = actions[b, action_idx, 3]  # 네 번째 요소 (파라미터)
                            targets.append(param_value)
                        else:
                            # 2D인 경우 전체 사용 (fallback)
                            targets.append(actions[b, action_idx])
        
        if len(targets) == 0:
            return None
            
        targets_tensor = torch.stack(targets).to(self.device)
        return F.mse_loss(param_logits, targets_tensor.float())
    
    def _extract_targets(self, data: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
        """SAR 시퀀스에서 타겟 추출"""
        targets = []
        batch_size, sar_seq_len = mask.shape
        _, data_seq_len = data.shape[:2]
        
        for b in range(batch_size):
            for sar_idx in range(sar_seq_len):
                if mask[b, sar_idx]:
                    action_idx = sar_idx // 3  # SAR -> Action 매핑
                    if action_idx < data_seq_len:
                        if len(data.shape) > 2:
                            # 다차원 데이터에서 첫 번째 요소만 추출 (스칼라)
                            if data.shape[2] == 1:
                                targets.append(data[b, action_idx, 0])
                            else:
                                # 벡터인 경우 첫 번째 요소만 (gate prediction용)
                                targets.append(data[b, action_idx, 0])
                        else:
                            # 2D 데이터는 그대로 사용
                            targets.append(data[b, action_idx])
        
        return targets
    
    def _log_metrics(self, loss_value: float):
        """메트릭 로깅"""
        if not self.use_wandb:
            return
            
        log_dict = {
            'train/loss': loss_value,
            'train/lr': self.optimizer.param_groups[0]['lr'],
            'train/step': self.global_step,
            'train/epoch': self.current_epoch
        }
        
        # 메모리 정보 추가
        memory_info = self.get_memory_status()
        if 'allocated_gb' in memory_info:
            log_dict['system/gpu_memory_gb'] = memory_info['allocated_gb']
            
        wandb.log(log_dict, step=self.global_step)
    
    def _prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """입력 텐서 준비"""
        return {
            'input_sequence': batch['input_sequence'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
            'action_prediction_mask': batch['action_prediction_mask'].to(self.device)
        }
    
    def _compute_loss(self, predictions: Dict[str, torch.Tensor], 
                     batch: Dict[str, torch.Tensor], 
                     mask: torch.Tensor) -> torch.Tensor:
        """손실 계산 - 단순화"""
        losses = []
        
        # Gate prediction loss
        if 'actions' in batch and batch['actions'] is not None:
            gate_loss = self._compute_gate_loss(predictions['gate'], batch['actions'], mask)
            if gate_loss is not None:
                losses.append(gate_loss)
        
        # Position prediction loss - actions에서 qubit 위치 추출
        if 'actions' in batch and batch['actions'] is not None:
            pos_loss = self._compute_position_loss(predictions['position'], batch['actions'], mask)
            if pos_loss is not None:
                losses.append(pos_loss)
        
        # Parameter prediction loss - actions에서 파라미터 추출
        if 'actions' in batch and batch['actions'] is not None:
            param_loss = self._compute_parameter_loss(predictions['parameter'], batch['actions'], mask)
            if param_loss is not None:
                losses.append(param_loss)
        
        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = str(Path(filepath).parent / 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"✅ Best model saved to {best_path}")
    
    def load_checkpoint(self, filepath: str):
        """체크포인트 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.global_step = checkpoint['global_step']
        
        print(f"✅ Checkpoint loaded from {filepath}")
        print(f"   - Epoch: {self.current_epoch}")
        print(f"   - Best Val Loss: {self.best_val_loss:.4f}")
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: str = "checkpoints"
    ):
        """전체 훈련 루프"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        print(f"🚀 Decision Transformer 훈련 시작")
        print(f"   - 에포크: {num_epochs}")
        print(f"   - 체크포인트 저장: {save_path}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 훈련
            train_loss = self.train_epoch(train_loader)
            
            # 검증
            val_loss = self.validate(val_loader)
            
            # 스케줄러 업데이트
            self.scheduler.step()
            
            # 결과 출력
            print(f"\n📊 Epoch {epoch+1}/{num_epochs}")
            print(f"   - Train Loss: {train_loss:.4f}")
            print(f"   - Val Loss: {val_loss:.4f}")
            print(f"   - LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # WandB 에포크 로깅 (상세)
            if self.use_wandb:
                epoch_log = {
                    'epoch': epoch,
                    'train/epoch_loss': train_loss,
                    'val/epoch_loss': val_loss,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'val/loss_improvement': self.best_val_loss - val_loss if val_loss < self.best_val_loss else 0,
                    'train/best_val_loss': self.best_val_loss
                }
                
                # 모델 파라미터 통계
                param_stats = self._get_model_parameter_stats()
                epoch_log.update(param_stats)
                
                wandb.log(epoch_log)
            
            # 체크포인트 저장
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            checkpoint_path = save_path / f"checkpoint_epoch_{epoch+1}.pt"
            self.save_checkpoint(str(checkpoint_path), is_best)
            
            # Early stopping (선택적)
            if hasattr(self.config, 'early_stopping_patience'):
                # 구현 가능
                pass
        
        print(f"🎉 훈련 완료! Best Val Loss: {self.best_val_loss:.4f}")
        
        if self.use_wandb:
            wandb.finish()
    
    def _smart_memory_cleanup(self, batch_idx: int):
        """효율적인 CUDA 메모리 정리"""
        # CUDA가 사용 가능하지 않으면 스킵
        if not torch.cuda.is_available():
            return
        
        # 정리 간격 및 메모리 임계값 확인
        should_cleanup = False
        
        # 1. 주기적 정리 (너무 자주 하지 않게)
        if batch_idx - self.last_cleanup_batch >= self.cleanup_interval:
            should_cleanup = True
            reason = f"interval ({self.cleanup_interval} batches)"
        
        # 2. 메모리 임계값 초과 시 정리
        if not should_cleanup:
            memory_info = get_memory_info()
            if 'allocated_gb' in memory_info and memory_info['allocated_gb'] > self.memory_threshold_gb:
                should_cleanup = True
                reason = f"memory threshold ({memory_info['allocated_gb']:.1f}GB > {self.memory_threshold_gb}GB)"
        
        # 메모리 정리 실행
        if should_cleanup:
            cleanup_memory()
            self.last_cleanup_batch = batch_idx
            
            # 디버그 정보 (가끔 출력)
            if batch_idx % (self.cleanup_interval * 2) == 0:  # 덜 빈번하게 출력
                memory_after = get_memory_info()
                if 'allocated_gb' in memory_after:
                    print(f"🧹 Memory cleanup at batch {batch_idx} ({reason}) - Memory: {memory_after['allocated_gb']:.1f}GB")
    
    def get_memory_status(self) -> Dict[str, Any]:
        """현재 메모리 상태 반환"""
        return get_memory_info()
    
    def force_memory_cleanup(self):
        """강제 메모리 정리"""
        cleanup_memory()
        print(f"🧹 Forced memory cleanup - Memory: {get_memory_info().get('allocated_gb', 'N/A')}GB")
    
    def _get_model_parameter_stats(self) -> Dict[str, float]:
        """모델 파라미터 통계 수집"""
        stats = {}
        
        # 전체 파라미터 통계
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        stats['model/total_parameters'] = total_params
        stats['model/trainable_parameters'] = trainable_params
        
        # 가중치 통계
        weights = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                weights.extend(param.data.flatten().cpu().numpy())
        
        if weights:
            import numpy as np
            weights = np.array(weights)
            stats['model/weight_mean'] = float(np.mean(weights))
            stats['model/weight_std'] = float(np.std(weights))
            stats['model/weight_min'] = float(np.min(weights))
            stats['model/weight_max'] = float(np.max(weights))
        
        # 그래디언트 통계
        grads = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads.extend(param.grad.data.flatten().cpu().numpy())
        
        if grads:
            import numpy as np
            grads = np.array(grads)
            stats['model/grad_mean'] = float(np.mean(grads))
            stats['model/grad_std'] = float(np.std(grads))
            stats['model/grad_min'] = float(np.min(grads))
            stats['model/grad_max'] = float(np.max(grads))
        
        return stats
