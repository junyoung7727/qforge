"""
Loss Tracking and Logging System
"""
import torch
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import matplotlib.pyplot as plt


class LossTracker:
    """손실 추적 및 로깅 시스템"""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss history storage
        self.train_losses = defaultdict(list)
        self.val_losses = defaultdict(list)
        self.epoch_times = []
        self.learning_rates = []
        
        # Current epoch tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Real-time logging
        self.log_file = self.save_dir / "training_log.txt"
        self.loss_history_file = self.save_dir / "loss_history.json"
        
    def log_epoch_start(self, epoch: int, lr: float):
        """에포크 시작 로깅"""
        self.current_epoch = epoch
        self.learning_rates.append(lr)
        
        message = f"\n{'='*60}\nEpoch {epoch} started - LR: {lr:.2e}\n{'='*60}"
        self._write_log(message)
        print(message)
        
    def log_train_losses(self, losses: Dict[str, float], batch_count: int):
        """훈련 손실 로깅"""
        # Average losses over batches
        avg_losses = {k: v / batch_count for k, v in losses.items()}
        
        # Store in history
        for key, value in avg_losses.items():
            self.train_losses[key].append(value)
        
        # Separate raw and weighted losses for cleaner logging
        raw_losses = {k: v for k, v in avg_losses.items() if not k.startswith('weighted_')}
        weighted_losses = {k.replace('weighted_', ''): v for k, v in avg_losses.items() if k.startswith('weighted_')}
        
        # Log raw losses
        raw_loss_str = " | ".join([f"{k}: {v:.6f}" for k, v in raw_losses.items()])
        message = f"Train Losses - {raw_loss_str}"
        self._write_log(message)
        print(f"📊 {message}")
        
        # Log weighted losses if available
        if weighted_losses:
            weighted_loss_str = " | ".join([f"{k}: {v:.6f}" for k, v in weighted_losses.items()])
            weighted_message = f"Train Weighted - {weighted_loss_str}"
            self._write_log(weighted_message)
            print(f"⚖️ {weighted_message}")
        
    def log_val_losses(self, losses: Dict[str, float], batch_count: int):
        """검증 손실 로깅"""
        # Average losses over batches
        avg_losses = {k: v / batch_count for k, v in losses.items()}
        
        # Store in history
        for key, value in avg_losses.items():
            self.val_losses[key].append(value)
        
        # Check for best validation loss
        current_val_loss = avg_losses.get('total', float('inf'))
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_epoch = self.current_epoch
            improvement_msg = f" 🎉 NEW BEST! (Improved by {self.best_val_loss - current_val_loss:.6f})"
        else:
            improvement_msg = f" (Best: {self.best_val_loss:.6f} at epoch {self.best_epoch})"
        
        # Separate raw and weighted losses for cleaner logging
        raw_losses = {k: v for k, v in avg_losses.items() if not k.startswith('weighted_')}
        weighted_losses = {k.replace('weighted_', ''): v for k, v in avg_losses.items() if k.startswith('weighted_')}
        
        # Log raw losses
        raw_loss_str = " | ".join([f"{k}: {v:.6f}" for k, v in raw_losses.items()])
        message = f"Val Losses - {raw_loss_str}{improvement_msg}"
        self._write_log(message)
        print(f"📈 {message}")
        
        # Log weighted losses if available
        if weighted_losses:
            weighted_loss_str = " | ".join([f"{k}: {v:.6f}" for k, v in weighted_losses.items()])
            weighted_message = f"Val Weighted - {weighted_loss_str}"
            self._write_log(weighted_message)
            print(f"⚖️ {weighted_message}")
        
    def log_epoch_end(self, epoch_time: float):
        """에포크 종료 로깅"""
        self.epoch_times.append(epoch_time)
        
        message = f"Epoch {self.current_epoch} completed in {epoch_time:.2f}s"
        self._write_log(message)
        print(f"⏱️ {message}")
        
        # Save loss history after each epoch
        self._save_loss_history()
        
    def log_batch_error(self, batch_idx: int, error: str, stage: str = "train"):
        """배치 에러 로깅"""
        message = f"ERROR - {stage.upper()} Batch {batch_idx}: {error}"
        self._write_log(message)
        print(f"❌ {message}")
        
    def log_gradient_info(self, batch_idx: int, grad_norm: float):
        """그래디언트 정보 로깅"""
        message = f"Batch {batch_idx} - Gradient norm: {grad_norm:.6f}"
        self._write_log(message)
        
    def log_memory_info(self, batch_idx: int, memory_info: Dict[str, Any]):
        """메모리 정보 로깅"""
        if memory_info:
            message = f"Batch {batch_idx} - GPU Memory: {memory_info.get('allocated_gb', 0):.2f}GB / {memory_info.get('total_gb', 0):.2f}GB"
            self._write_log(message)
            
    def log_perfect_fidelity_skip(self, batch_idx: int, count: int):
        """완벽한 피델리티 배치의 피델리티 손실만 스킵 로깅"""
        message = f"Batch {batch_idx} - Fidelity Loss Zeroed (perfect fidelity batch #{count}/1)"
        self._write_log(message)
        #print(f"🔇 {message}")
        
    def create_loss_plots(self):
        """손실 그래프 생성"""
        if not self.train_losses or not self.val_losses:
            return
            
        # Create plots directory
        plots_dir = self.save_dir / "loss_plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Plot each loss component
        for loss_key in self.train_losses.keys():
            if loss_key in self.val_losses:
                plt.figure(figsize=(10, 6))
                
                epochs = range(1, len(self.train_losses[loss_key]) + 1)
                plt.plot(epochs, self.train_losses[loss_key], 'b-', label=f'Train {loss_key}', alpha=0.8)
                plt.plot(epochs, self.val_losses[loss_key], 'r-', label=f'Val {loss_key}', alpha=0.8)
                
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'{loss_key.title()} Loss Over Time')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.yscale('log' if loss_key != 'total' else 'linear')
                
                plt.savefig(plots_dir / f"{loss_key}_loss.png", dpi=300, bbox_inches='tight')
                plt.close()
                
        print(f"📊 Loss plots saved to {plots_dir}")
        
    def get_summary(self) -> Dict[str, Any]:
        """학습 요약 정보 반환"""
        return {
            'total_epochs': len(self.train_losses.get('total', [])),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'total_training_time': sum(self.epoch_times),
            'avg_epoch_time': sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0,
            'final_train_loss': self.train_losses.get('total', [0])[-1] if self.train_losses.get('total') else 0,
            'final_val_loss': self.val_losses.get('total', [0])[-1] if self.val_losses.get('total') else 0
        }
        
    def _write_log(self, message: str):
        """로그 파일에 메시지 작성"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
            
    def _save_loss_history(self):
        """손실 히스토리를 JSON 파일로 저장"""
        history_data = {
            'train_losses': dict(self.train_losses),
            'val_losses': dict(self.val_losses),
            'epoch_times': self.epoch_times,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'current_epoch': self.current_epoch
        }
        
        with open(self.loss_history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
