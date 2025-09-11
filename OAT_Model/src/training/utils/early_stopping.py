"""
Early Stopping Implementation
학습 중 조기 종료를 위한 유틸리티
"""

import torch
from typing import Dict, Any, Optional


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in the monitored quantity to qualify as an improvement
            restore_best_weights: Whether to restore model weights from the epoch with the best value
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_score = None
        self.early_stopped = False
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: Optional[torch.nn.Module] = None) -> bool:
        """
        Check if training should be stopped
        
        Args:
            val_loss: Current validation loss
            model: Model to save best weights (optional)
            
        Returns:
            True if training should be stopped, False otherwise
        """
        score = -val_loss  # Convert loss to score (higher is better)
        
        if self.best_score is None:
            self.best_score = score
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stopped = True
                if model is not None and self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        
        return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for checkpointing"""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stopped': self.early_stopped,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'restore_best_weights': self.restore_best_weights
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from checkpoint"""
        self.counter = state.get('counter', 0)
        self.best_score = state.get('best_score', None)
        self.early_stopped = state.get('early_stopped', False)
        self.patience = state.get('patience', self.patience)
        self.min_delta = state.get('min_delta', self.min_delta)
        self.restore_best_weights = state.get('restore_best_weights', self.restore_best_weights)
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stopped = False
        self.best_weights = None


class EarlyStoppingWithWarmup(EarlyStopping):
    """Early stopping with warmup period"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, 
                 restore_best_weights: bool = True, warmup_epochs: int = 0):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in the monitored quantity to qualify as an improvement
            restore_best_weights: Whether to restore model weights from the epoch with the best value
            warmup_epochs: Number of epochs to wait before starting early stopping
        """
        super().__init__(patience, min_delta, restore_best_weights)
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def __call__(self, val_loss: float, model: Optional[torch.nn.Module] = None) -> bool:
        """
        Check if training should be stopped (with warmup)
        
        Args:
            val_loss: Current validation loss
            model: Model to save best weights (optional)
            
        Returns:
            True if training should be stopped, False otherwise
        """
        self.current_epoch += 1
        
        # Don't apply early stopping during warmup
        if self.current_epoch <= self.warmup_epochs:
            # Still track best score during warmup
            score = -val_loss
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                if model is not None and self.restore_best_weights:
                    self.best_weights = model.state_dict().copy()
            return False
        
        # Apply normal early stopping after warmup
        return super().__call__(val_loss, model)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for checkpointing"""
        state = super().get_state()
        state['warmup_epochs'] = self.warmup_epochs
        state['current_epoch'] = self.current_epoch
        return state
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from checkpoint"""
        super().load_state(state)
        self.warmup_epochs = state.get('warmup_epochs', self.warmup_epochs)
        self.current_epoch = state.get('current_epoch', 0)
    
    def reset(self):
        """Reset early stopping state"""
        super().reset()
        self.current_epoch = 0


if __name__ == "__main__":
    # Test early stopping
    import torch.nn as nn
    
    # Create dummy model
    model = nn.Linear(10, 1)
    
    # Test basic early stopping
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)
    
    # Simulate training with decreasing then increasing loss
    val_losses = [1.0, 0.8, 0.6, 0.65, 0.67, 0.69, 0.70]  # Should stop at index 6
    
    print("Testing EarlyStopping:")
    for i, loss in enumerate(val_losses):
        should_stop = early_stopping(loss, model)
        print(f"Epoch {i}: loss={loss:.3f}, counter={early_stopping.counter}, should_stop={should_stop}")
        if should_stop:
            break
    
    print(f"Early stopped: {early_stopping.early_stopped}")
    
    # Test with warmup
    print("\nTesting EarlyStoppingWithWarmup:")
    early_stopping_warmup = EarlyStoppingWithWarmup(patience=2, min_delta=0.01, warmup_epochs=3)
    
    for i, loss in enumerate(val_losses):
        should_stop = early_stopping_warmup(loss, model)
        print(f"Epoch {i}: loss={loss:.3f}, counter={early_stopping_warmup.counter}, should_stop={should_stop}")
        if should_stop:
            break
