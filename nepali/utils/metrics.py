"""
Metrics for ASR evaluation
"""

import editdistance
import torch


def calculate_cer(predictions, references):
    """
    Calculate Character Error Rate.
    
    Args:
        predictions: List of predicted text strings
        references: List of reference text strings
    Returns:
        CER as percentage
    """
    total_chars = 0
    total_errors = 0
    
    for pred, ref in zip(predictions, references):
        # Remove spaces for CER calculation
        pred_chars = pred.replace(' ', '')
        ref_chars = ref.replace(' ', '')
        
        if len(ref_chars) == 0:
            continue
        
        errors = editdistance.eval(pred_chars, ref_chars)
        total_errors += errors
        total_chars += len(ref_chars)
    
    return (total_errors / total_chars) * 100 if total_chars > 0 else 0.0


def calculate_wer(predictions, references):
    """
    Calculate Word Error Rate.
    
    Args:
        predictions: List of predicted text strings
        references: List of reference text strings
    Returns:
        WER as percentage
    """
    total_words = 0
    total_errors = 0
    
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref.split()
        
        if len(ref_words) == 0:
            continue
        
        errors = editdistance.eval(pred_words, ref_words)
        total_errors += errors
        total_words += len(ref_words)
    
    return (total_errors / total_words) * 100 if total_words > 0 else 0.0


class MetricsTracker:
    """
    Track and compute metrics during training.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.references = []
        self.losses = []
    
    def update(self, preds, refs, loss=None):
        """
        Update metrics with new predictions.
        
        Args:
            preds: List of predicted text strings
            refs: List of reference text strings
            loss: Optional loss value
        """
        self.predictions.extend(preds)
        self.references.extend(refs)
        
        if loss is not None:
            self.losses.append(loss)
    
    def compute(self):
        """
        Compute all metrics.
        
        Returns:
            Dictionary with CER, WER, and average loss
        """
        metrics = {
            'cer': calculate_cer(self.predictions, self.references),
            'wer': calculate_wer(self.predictions, self.references)
        }
        
        if self.losses:
            metrics['loss'] = sum(self.losses) / len(self.losses)
        
        return metrics
    
    def __str__(self):
        """String representation of current metrics."""
        metrics = self.compute()
        return f"CER: {metrics['cer']:.2f}%, WER: {metrics['wer']:.2f}%"