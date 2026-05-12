"""
Conformer Block and Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import (
    MultiHeadSelfAttentionModule,
    ConvolutionModule,
    FeedForwardModule,
    RelativePositionalEncoding
)


class ConformerBlock(nn.Module):
    """
    Conformer Block:
    - Feed-Forward Module (1/2 step)
    - Multi-Head Self-Attention Module
    - Convolution Module
    - Feed-Forward Module (1/2 step)
    - Layer Norm
    """
    def __init__(
        self,
        d_model,
        num_attention_heads,
        feed_forward_expansion_factor=4,
        conv_expansion_factor=2,
        conv_kernel_size=32,
        dropout=0.1
    ):
        super().__init__()
        
        # First Feed-Forward with half-step residual
        self.ffn1 = FeedForwardModule(
            d_model, 
            expansion_factor=feed_forward_expansion_factor,
            dropout=dropout
        )
        
        # Multi-Head Self-Attention
        self.mhsa = MultiHeadSelfAttentionModule(
            d_model,
            num_attention_heads,
            dropout=dropout
        )
        
        # Convolution Module
        self.conv = ConvolutionModule(
            d_model,
            kernel_size=conv_kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout=dropout
        )
        
        # Second Feed-Forward with half-step residual
        self.ffn2 = FeedForwardModule(
            d_model,
            expansion_factor=feed_forward_expansion_factor,
            dropout=dropout
        )
        
        # Final Layer Norm
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, pos_emb, mask=None):
        """
        Args:
            x: (batch, time, d_model)
            pos_emb: (batch, time, d_model)
            mask: (batch, time)
        Returns:
            (batch, time, d_model)
        """
        # First FFN with half-step residual
        x = x + 0.5 * self.ffn1(x)
        
        # Multi-Head Self-Attention
        x = self.mhsa(x, pos_emb, mask)
        
        # Convolution
        x = self.conv(x)
        
        # Second FFN with half-step residual
        x = x + 0.5 * self.ffn2(x)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        return x


class ConvSubsampling(nn.Module):
    """
    Convolutional subsampling layer to reduce sequence length.
    Reduces time dimension by factor of 4.
    """
    def __init__(self, in_channels, out_channels, input_dim=80):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2)
        
        # Calculate output dimension after convolutions
        # After conv1: (input_dim - 1) // 2
        # After conv2: ((input_dim - 1) // 2 - 1) // 2
        output_dim = ((input_dim - 1) // 2 - 1) // 2
        self.linear = nn.Linear(out_channels * output_dim, out_channels)
    
    def forward(self, x):
        """
        Args:
            x: (batch, time, n_mels)
        Returns:
            (batch, time // 4, out_channels)
        """
        # Add channel dimension: (batch, 1, time, n_mels)
        x = x.unsqueeze(1)
        
        # Convolution layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Reshape: (batch, channels, time, freq) -> (batch, time, channels * freq)
        batch, channels, time, freq = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch, time, channels * freq)
        
        # Linear projection
        x = self.linear(x)
        
        return x


class ConformerEncoder(nn.Module):
    """
    Conformer Encoder with multiple Conformer blocks.
    """
    def __init__(
        self,
        input_dim=80,
        encoder_dim=256,
        num_layers=16,
        num_attention_heads=4,
        feed_forward_expansion_factor=4,
        conv_expansion_factor=2,
        conv_kernel_size=32,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        
        # Convolutional subsampling
        self.subsampling = ConvSubsampling(1, encoder_dim, input_dim=input_dim)
        
        # Positional encoding
        self.pos_encoding = RelativePositionalEncoding(encoder_dim, max_len)
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=encoder_dim,
                num_attention_heads=num_attention_heads,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, time, n_mels) - mel spectrogram features
            lengths: (batch,) - actual sequence lengths before padding
        Returns:
            x: (batch, time // 4, encoder_dim)
            lengths: (batch,) - updated sequence lengths
        """
        # Convolutional subsampling
        x = self.subsampling(x)
        x = self.dropout(x)
        
        # Update lengths after subsampling
        if lengths is not None:
            lengths = ((lengths - 1) // 2 - 1) // 2
        
        # Create padding mask
        mask = None
        if lengths is not None:
            batch_size, max_len, _ = x.size()
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        # Get positional encoding
        pos_emb = self.pos_encoding(x)
        
        # Pass through Conformer blocks
        for block in self.conformer_blocks:
            x = block(x, pos_emb, mask)
        
        return x, lengths