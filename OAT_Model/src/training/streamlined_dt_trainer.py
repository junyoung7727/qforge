"""
Streamlined Decision Transformer Trainer
간소화된 훈련 로직 - 불필요한 복잡성 제거
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, Any, Optional
import wandb
from tqdm import tqdm


class StreamlinedDTTrainer:
    """간소화된 Decision Transformer 훈련기"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        vocab_size: int,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01
    ):
        self.model = model
        self.vocab_size = vocab_size
        self.device = torch.device(device)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.global_step = 0
        
    def train_epoch(self, train_loader) -> float:
        """훈련 에포크 실행"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            loss = self._train_step(batch)
            if loss is not None:
                total_loss += loss
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss:.4f}'})
                
                # Log to wandb
                if wandb.run is not None:
                    wandb.log({
                        'train/step_loss': loss,
                        'train/learning_rate': self.optimizer.param_groups[0]['lr']
                    }, step=self.global_step)
                
                self.global_step += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Optional[float]:
        """단일 훈련 스텝"""
        try:
            # Move batch to device
            input_sequence = batch['input_sequence'].to(self.device)
            action_targets = batch['action_targets'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                input_ids=input_sequence,
                attention_mask=attention_mask
            )
            
            # Extract logits for action prediction positions
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # Calculate loss only for action positions (every 3rd position starting from 1)
            loss = self._compute_action_loss(logits, action_targets, attention_mask)
            
            if loss is None:
                return None
                
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            print(f"❌ Training step failed: {e}")
            return None
    
    def _compute_action_loss(
        self, 
        logits: torch.Tensor, 
        action_targets: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """액션 예측 손실 계산"""
        batch_size, seq_len, _ = logits.shape
        _, num_actions = action_targets.shape
        
        # Extract action positions (positions 1, 4, 7, ... in SAR sequence)
        action_logits = []
        action_labels = []
        
        for b in range(batch_size):
            for a in range(num_actions):
                action_pos = a * 3 + 1  # Action positions in SAR pattern
                if action_pos < seq_len and attention_mask[b, action_pos] > 0:
                    action_logits.append(logits[b, action_pos])
                    
                    # Clamp target to valid vocabulary range
                    target = action_targets[b, a].item()
                    target = max(0, min(target, self.vocab_size - 1))
                    action_labels.append(target)
        
        if not action_logits:
            return None
            
        # Stack and compute cross entropy loss
        action_logits = torch.stack(action_logits)
        action_labels = torch.tensor(action_labels, device=self.device, dtype=torch.long)
        
        return F.cross_entropy(action_logits, action_labels)
    
    def validate(self, val_loader) -> Dict[str, float]:
        """검증 실행"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                try:
                    input_sequence = batch['input_sequence'].to(self.device)
                    action_targets = batch['action_targets'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_sequence,
                        attention_mask=attention_mask
                    )
                    
                    loss = self._compute_action_loss(
                        outputs.logits, action_targets, attention_mask
                    )
                    
                    if loss is not None:
                        total_loss += loss.item()
                        num_batches += 1
                        
                except Exception as e:
                    print(f"❌ Validation step failed: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        return {
            'val_loss': avg_loss,
            'val_batches': num_batches
        }
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        save_path: str,
        validate_every: int = 1
    ):
        """전체 훈련 루프"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n🚀 Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            print(f"   Train Loss: {train_loss:.4f}")
            
            # Validation
            if (epoch + 1) % validate_every == 0:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['val_loss']
                print(f"   Val Loss: {val_loss:.4f}")
                
                # Log to wandb
                if wandb.run is not None:
                    wandb.log({
                        'train/epoch_loss': train_loss,
                        'val/epoch_loss': val_loss,
                        'epoch': epoch + 1
                    }, step=self.global_step)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(save_path)
                    print(f"   💾 Best model saved (val_loss: {val_loss:.4f})")
            else:
                # Log training only
                if wandb.run is not None:
                    wandb.log({
                        'train/epoch_loss': train_loss,
                        'epoch': epoch + 1
                    }, step=self.global_step)
    
    def save_model(self, path: str):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'vocab_size': self.vocab_size
        }, path)
    
    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        print(f"✅ Model loaded from {path}")
