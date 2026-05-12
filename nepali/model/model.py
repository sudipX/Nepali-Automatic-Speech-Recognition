"""
Complete Conformer ASR Model with CTC Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conformer import ConformerEncoder


class ConformerCTC(nn.Module):
    """
    Conformer model with CTC loss for speech recognition.
    """
    def __init__(
        self,
        vocab_size,
        input_dim=80,
        encoder_dim=256,
        num_encoder_layers=16,
        num_attention_heads=4,
        feed_forward_expansion_factor=4,
        conv_expansion_factor=2,
        conv_kernel_size=32,
        dropout=0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        
        # Conformer encoder
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout
        )
        
        # CTC projection layer
        self.ctc_projection = nn.Linear(encoder_dim, vocab_size)
    
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: (batch, time, n_mels)
            lengths: (batch,) - sequence lengths
        Returns:
            log_probs: (batch, time // 4, vocab_size)
            lengths: (batch,) - updated lengths after subsampling
        """
        # Encode
        x, lengths = self.encoder(x, lengths)
        
        # CTC projection
        logits = self.ctc_projection(x)
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs, lengths
    
    def compute_loss(self, log_probs, targets, input_lengths, target_lengths):
        """
        Compute CTC loss.
        
        Args:
            log_probs: (batch, time, vocab_size)
            targets: (batch, max_target_length)
            input_lengths: (batch,)
            target_lengths: (batch,)
        Returns:
            loss: scalar
        """
        # CTC expects (time, batch, vocab_size)
        log_probs = log_probs.transpose(0, 1)
        
        loss = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction='mean',
            zero_infinity=True
        )
        
        return loss
    
    def decode(self, log_probs, lengths=None):
        """
        Greedy decoding for inference.
        
        Args:
            log_probs: (batch, time, vocab_size)
            lengths: (batch,) - optional sequence lengths
        Returns:
            predictions: list of predicted token sequences
        """
        # Greedy decoding
        predictions = log_probs.argmax(dim=-1)  # (batch, time)
        
        decoded = []
        for i, pred in enumerate(predictions):
            if lengths is not None:
                pred = pred[:lengths[i]]
            
            # Remove consecutive duplicates and blanks
            pred = pred.cpu().tolist()
            decoded_seq = []
            prev_token = None
            
            for token in pred:
                if token != 0 and token != prev_token:  # 0 is blank
                    decoded_seq.append(token)
                prev_token = token
            
            decoded.append(decoded_seq)
        
        return decoded
    
    @torch.no_grad()
    def recognize(self, x, lengths=None):
        """
        Run inference on audio features.
        
        Args:
            x: (batch, time, n_mels)
            lengths: (batch,)
        Returns:
            predictions: list of decoded sequences
        """
        self.eval()
        log_probs, lengths = self.forward(x, lengths)
        predictions = self.decode(log_probs, lengths)
        return predictions


def build_model(config):
    """
    Build Conformer model from config.
    
    Args:
        config: dict with model configuration
    Returns:
        ConformerCTC model
    """
    model = ConformerCTC(
        vocab_size=config['tokenizer']['vocab_size'] + 1,  # +1 for blank token
        input_dim=config['audio']['n_mels'],
        encoder_dim=config['model']['encoder_dim'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_attention_heads=config['model']['num_attention_heads'],
        feed_forward_expansion_factor=config['model']['feed_forward_expansion_factor'],
        conv_expansion_factor=config['model']['conv_expansion_factor'],
        conv_kernel_size=config['model']['conv_kernel_size'],
        dropout=config['model']['dropout']
    )
    
    return model