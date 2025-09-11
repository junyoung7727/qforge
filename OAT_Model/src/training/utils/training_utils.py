"""
Training Utilities Module

Early stopping, 메모리 관리, WandB 초기화 등 학습 유틸리티 모듈
"""
import torch
import time
from typing import Optional, Dict


class EarlyStopping:
    """Early stopping 관리자"""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stopped = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Early stopping 체크
        
        Args:
            val_loss: 현재 검증 손실
            
        Returns:
            bool: True if should stop, False otherwise
        """
        improved = val_loss < self.best_loss - self.min_delta
        
        if improved:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stopped = True
                return True
            return False
    
    def reset(self):
        """Early stopping 상태 리셋"""
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stopped = False
    
    def get_state(self) -> dict:
        """현재 상태 반환"""
        return {
            'counter': self.counter,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'best_loss': self.best_loss,
            'early_stopped': self.early_stopped
        }
    
    def load_state(self, state: dict):
        """상태 복원"""
        self.counter = state.get('counter', 0)
        self.patience = state.get('patience', 15)
        self.min_delta = state.get('min_delta', 0.001)
        self.best_loss = state.get('best_loss', float('inf'))
        self.early_stopped = state.get('early_stopped', False)


class MemoryManager:
    """GPU 메모리 관리자"""
    
    def __init__(self, device: torch.device, cleanup_frequency: int = 10):
        self.device = device
        self.cleanup_frequency = cleanup_frequency
        self.step_count = 0
    
    def setup_gpu_optimization(self):
        """GPU 메모리 최적화 설정"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()  # 초기 메모리 정리
            torch.backends.cudnn.benchmark = True  # cuDNN 최적화
            torch.backends.cudnn.deterministic = False  # 성능 우선
            
            gpu_props = torch.cuda.get_device_properties(0)
            total_memory = gpu_props.total_memory / (1024**3)
            print(f"[GPU] 메모리 최적화 활성화 - 사용 가능: {total_memory:.1f}GB")
    
    def cleanup_if_needed(self, force: bool = False):
        """필요시 메모리 정리"""
        self.step_count += 1
        
        if force or (self.device.type == 'cuda' and self.step_count % self.cleanup_frequency == 0):
            torch.cuda.empty_cache()
    
    def get_memory_info(self) -> Optional[dict]:
        """GPU 메모리 사용량 정보 반환"""
        if self.device.type == 'cuda':
            return {
                'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3)
            }
        return None


class WandBManager:
    """Weights & Biases 관리자"""
    
    def __init__(self, config, use_wandb: bool = True):
        self.config = config
        self.use_wandb = use_wandb
        self.initialized = False
    
    def init_wandb(self, project_name: str, run_name: str, config_dict: dict, run_id: str) -> bool:
        """Initialize Weights & Biases logging"""
        if not self.use_wandb:
            return False
            
        import wandb
        
        wandb.init(
            project=project_name,
            name=run_name,
            config=config_dict,
            resume="allow",
            id=run_id
        )
        
        self.initialized = True
        print(f"✅ WandB initialized: {project_name}/{run_name}")
        return True
    
    def log(self, metrics: dict):
        """메트릭 로깅"""
        if self.use_wandb and self.initialized:
            import wandb
            wandb.log(metrics)
    
    def log_loss_tracking(self, train_losses: dict, val_losses: dict, epoch: int, lr: float):
        """손실 추적 전용 로깅"""
        if self.use_wandb and self.initialized:
            import wandb
            
            # Create comprehensive loss tracking
            loss_metrics = {
                'epoch': epoch,
                'learning_rate': lr,
                
                # Training losses (raw)
                'train/total_loss': train_losses.get('total', 0),
                'train/entanglement_loss': train_losses.get('entanglement', 0),
                'train/fidelity_loss': train_losses.get('fidelity', 0),
                'train/expressibility_loss': train_losses.get('expressibility', 0),
                'train/combined_loss': train_losses.get('combined', 0),
                
                # Training weighted losses
                'train_weighted/entanglement_loss': train_losses.get('weighted_entanglement', 0),
                'train_weighted/fidelity_loss': train_losses.get('weighted_fidelity', 0),
                'train_weighted/expressibility_loss': train_losses.get('weighted_expressibility', 0),
                'train_weighted/combined_loss': train_losses.get('weighted_combined', 0),
                
                # Validation losses (raw)
                'val/total_loss': val_losses.get('total', 0),
                'val/entanglement_loss': val_losses.get('entanglement', 0),
                'val/fidelity_loss': val_losses.get('fidelity', 0),
                'val/expressibility_loss': val_losses.get('expressibility', 0),
                'val/combined_loss': val_losses.get('combined', 0),
                
                # Validation weighted losses
                'val_weighted/entanglement_loss': val_losses.get('weighted_entanglement', 0),
                'val_weighted/fidelity_loss': val_losses.get('weighted_fidelity', 0),
                'val_weighted/expressibility_loss': val_losses.get('weighted_expressibility', 0),
                'val_weighted/combined_loss': val_losses.get('weighted_combined', 0),
                
                # Loss ratios for analysis
                'analysis/train_val_ratio': train_losses.get('total', 1) / max(val_losses.get('total', 1), 1e-8),
                'analysis/entanglement_dominance': train_losses.get('weighted_entanglement', 0) / max(train_losses.get('total', 1), 1e-8),
                'analysis/fidelity_dominance': train_losses.get('weighted_fidelity', 0) / max(train_losses.get('total', 1), 1e-8),
                'analysis/expressibility_dominance': train_losses.get('weighted_expressibility', 0) / max(train_losses.get('total', 1), 1e-8),
                'analysis/combined_dominance': train_losses.get('weighted_combined', 0) / max(train_losses.get('total', 1), 1e-8),
            }
            
            wandb.log(loss_metrics)
    
    def finish(self):
        """WandB 세션 종료"""
        if self.use_wandb and self.initialized:
            import wandb
            wandb.finish()


class GradientManager:
    """그래디언트 관리자"""
    
    def __init__(self, max_norm: float = 0.5):
        self.max_norm = max_norm
    
    def check_and_clip_gradients(self, model: torch.nn.Module, batch_idx: int):
        """그래디언트 NaN 체크 및 클리핑"""
        # 그래디언트 NaN 체크 및 복구 시도
        nan_detected = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"Warning: NaN/Inf gradient detected in {name} at batch {batch_idx}")
                    # NaN/Inf 그래디언트를 0으로 대체
                    param.grad[torch.isnan(param.grad)] = 0.0
                    param.grad[torch.isinf(param.grad)] = 0.0
                    nan_detected = True
        
        if nan_detected:
            print(f"Batch {batch_idx}: NaN/Inf gradients replaced with zeros")
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)


class LossValidator:
    """손실 값 검증자"""
    
    @staticmethod
    def validate_loss(loss: torch.Tensor, batch_idx: int, stage: str = "train"):
        """손실 값 NaN/Inf 체크"""
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"{stage} 배치 {batch_idx}: NaN/Inf loss 감지")
    
    @staticmethod
    def validate_predictions(predictions: Dict[str, torch.Tensor], batch_idx: int, stage: str = "train"):
        """예측값 NaN/Inf 체크"""
        for key, pred in predictions.items():
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                raise RuntimeError(f"{stage} 배치 {batch_idx}: {key}에서 NaN/Inf 예측값 감지")


class TrainingTimer:
    """학습 시간 측정자"""
    
    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None
    
    def start_training(self):
        """전체 학습 시작"""
        self.start_time = time.time()
    
    def start_epoch(self):
        """에폭 시작"""
        self.epoch_start_time = time.time()
    
    def get_epoch_duration(self) -> float:
        """현재 에폭 소요 시간 반환"""
        if self.epoch_start_time is None:
            return 0.0
        return time.time() - self.epoch_start_time
    
    def get_epoch_time(self) -> float:
        """에폭 시간 반환 (별칭)"""
        return self.get_epoch_duration()
    
    def get_total_duration(self) -> float:
        """전체 학습 소요 시간 반환"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def format_duration(self, duration: float) -> str:
        """시간을 시:분:초 형식으로 포맷"""
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
