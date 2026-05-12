"""
Data Augmentation for Nepali Speech Recognition
Includes SpecAugment and other techniques
"""

import torch
import torch.nn as nn
import random


class SpecAugment(nn.Module):
    """
    SpecAugment: A Simple Data Augmentation Method for ASR
    Based on: https://arxiv.org/abs/1904.08779
    
    Optimized for Nepali ASR with parameters from Conformer paper.
    """
    def __init__(
        self, 
        freq_mask_param=27,
        time_mask_param=0.05,
        n_freq_masks=2,
        n_time_masks=10,
        p=1.0
    ):
        """
        Args:
            freq_mask_param: Maximum width of frequency mask
            time_mask_param: Maximum ratio of time mask (relative to sequence length)
            n_freq_masks: Number of frequency masks to apply
            n_time_masks: Number of time masks to apply
            p: Probability of applying augmentation
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p
    
    def forward(self, mel_spec):
        """
        Apply SpecAugment to mel spectrogram.
        
        Args:
            mel_spec: (batch, time, freq) or (time, freq)
        Returns:
            Augmented mel_spec with same shape
        """
        if random.random() > self.p:
            return mel_spec
        
        # Handle both batched and unbatched input
        if len(mel_spec.shape) == 2:
            mel_spec = mel_spec.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        batch_size, time_steps, n_mels = mel_spec.shape
        mel_spec = mel_spec.clone()
        
        # Apply frequency masking
        for _ in range(self.n_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, n_mels))
            f0 = random.randint(0, n_mels - f)
            mel_spec[:, :, f0:f0+f] = 0
        
        # Apply time masking
        max_time_mask = int(self.time_mask_param * time_steps)
        for _ in range(self.n_time_masks):
            t = random.randint(0, min(max_time_mask, time_steps))
            if t > 0:
                t0 = random.randint(0, max(1, time_steps - t))
                mel_spec[:, t0:t0+t, :] = 0
        
        if squeeze:
            mel_spec = mel_spec.squeeze(0)
        
        return mel_spec


class TimeStretch(nn.Module):
    """
    Time stretching augmentation (speed perturbation).
    """
    def __init__(self, rates=[0.9, 1.0, 1.1], p=0.5):
        """
        Args:
            rates: List of stretching rates
            p: Probability of applying augmentation
        """
        super().__init__()
        self.rates = rates
        self.p = p
    
    def forward(self, mel_spec):
        """
        Apply time stretching.
        
        Args:
            mel_spec: (time, freq) or (batch, time, freq)
        """
        if random.random() > self.p:
            return mel_spec
        
        rate = random.choice(self.rates)
        if rate == 1.0:
            return mel_spec
        
        # Interpolate along time dimension
        if len(mel_spec.shape) == 2:
            mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)
            squeeze = True
        else:
            mel_spec = mel_spec.unsqueeze(1)
            squeeze = False
        
        new_length = int(mel_spec.size(2) * rate)
        stretched = torch.nn.functional.interpolate(
            mel_spec,
            size=(new_length, mel_spec.size(3)),
            mode='bilinear',
            align_corners=False
        )
        
        if squeeze:
            stretched = stretched.squeeze(0).squeeze(0)
        else:
            stretched = stretched.squeeze(1)
        
        return stretched


def build_augmentation(config):
    """
    Build augmentation pipeline from config.
    
    Args:
        config: Configuration dictionary
    Returns:
        Augmentation module or None
    """
    aug_config = config.get('augmentation', {})
    
    # SpecAugment (most important for Nepali ASR)
    if aug_config.get('spec_augment', {}).get('enabled', True):
        spec_aug_config = aug_config['spec_augment']
        return SpecAugment(
            freq_mask_param=spec_aug_config.get('freq_mask_param', 27),
            time_mask_param=spec_aug_config.get('time_mask_param', 0.05),
            n_freq_masks=spec_aug_config.get('n_freq_masks', 2),
            n_time_masks=spec_aug_config.get('n_time_masks', 10),
            p=spec_aug_config.get('p', 1.0)
        )
    
    return None