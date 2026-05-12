"""
Learning rate schedulers for Transformer models
"""

import math
import torch.optim as optim


class TransformerLRScheduler:
    """
    Transformer learning rate schedule with warmup.
    
    LR = scale * d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    
    This scheduler:
    1. Increases LR linearly during warmup
    2. Decreases LR by inverse square root after warmup
    """
    def __init__(self, optimizer, d_model, warmup_steps=10000, scale=1.0):
        """
        Args:
            optimizer: PyTorch optimizer
            d_model: Model dimension
            warmup_steps: Number of warmup steps
            scale: Scaling factor for learning rate
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.scale = scale
        self.step_num = 0
        self.current_lr = 0
        
    def step(self):
        """Update learning rate for current step."""
        self.step_num += 1
        lr = self.scale * (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_lr = lr
        return lr
    
    def get_last_lr(self):
        """Get current learning rate."""
        return [self.current_lr]
    
    def state_dict(self):
        """Return state dictionary."""
        return {
            'step_num': self.step_num,
            'current_lr': self.current_lr
        }
    
    def load_state_dict(self, state_dict):
        """Load state dictionary."""
        self.step_num = state_dict['step_num']
        self.current_lr = state_dict['current_lr']


class WarmupLRScheduler:
    """
    Simple warmup scheduler with cosine annealing.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_num = 0
        
    def step(self):
        """Update learning rate."""
        self.step_num += 1
        
        if self.step_num < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.step_num / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        """Get current learning rate."""
        return [self.optimizer.param_groups[0]['lr']]


def create_scheduler(optimizer, config, total_steps=None):
    """
    Create learning rate scheduler from config.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary
        total_steps: Total training steps (required for some schedulers)
    Returns:
        Learning rate scheduler
    """
    scheduler_type = config['training'].get('scheduler', 'transformer')
    warmup_steps = config['training'].get('warmup_steps', 5000)
    
    if scheduler_type == 'transformer':
        return TransformerLRScheduler(
            optimizer,
            d_model=config['model']['encoder_dim'],
            warmup_steps=warmup_steps,
            scale=config['training'].get('lr_scale', 1.0)
        )
    elif scheduler_type == 'warmup_cosine':
        if total_steps is None:
            raise ValueError("total_steps required for warmup_cosine scheduler")
        return WarmupLRScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=config['training'].get('min_lr', 1e-6)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")