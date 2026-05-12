"""
Conformer Model Modules
Based on: "Conformer: Convolution-augmented Transformer for Speech Recognition"
https://arxiv.org/abs/2005.08100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-headed attention with relative positional encoding.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # Linear projections
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.pos_encoding = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Learnable bias for content and position
        self.u_bias = nn.Parameter(torch.Tensor(num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(num_heads, self.d_head))
        
        # Initialize
        nn.init.xavier_uniform_(self.u_bias)
        nn.init.xavier_uniform_(self.v_bias)
    
    def forward(self, x, pos_emb, mask=None):
        """
        Args:
            x: (batch, time, d_model)
            pos_emb: (batch, time, d_model) or (1, time, d_model)
            mask: (batch, time) - True for positions to mask
        Returns:
            (batch, time, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections and reshape
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        p = self.pos_encoding(pos_emb).view(batch_size, seq_len, self.num_heads, self.d_head)
        
        # Transpose to (batch, num_heads, seq_len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        p = p.transpose(1, 2)
        
        # Content-based attention
        q_with_u = q + self.u_bias.unsqueeze(1)  # (batch, heads, 1, d_head)
        content_score = torch.matmul(q_with_u, k.transpose(-2, -1))
        
        # Position-based attention
        q_with_v = q + self.v_bias.unsqueeze(1)
        pos_score = torch.matmul(q_with_v, p.transpose(-2, -1))
        pos_score = self._relative_shift(pos_score)
        
        # Combine scores
        scores = (content_score + pos_score) / math.sqrt(self.d_head)
        
        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, time)
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(output)
    
    def _relative_shift(self, pos_score):
        """Shift the relative position scores."""
        batch_size, num_heads, seq_len1, seq_len2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_len1, 1)
        padded = torch.cat([zeros, pos_score], dim=-1)
        padded = padded.view(batch_size, num_heads, seq_len2 + 1, seq_len1)
        shifted = padded[:, :, 1:].view_as(pos_score)
        return shifted


class MultiHeadSelfAttentionModule(nn.Module):
    """
    Multi-Head Self-Attention Module with pre-norm and residual connection.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, pos_emb, mask=None):
        """
        Args:
            x: (batch, time, d_model)
            pos_emb: (batch, time, d_model)
            mask: (batch, time)
        Returns:
            (batch, time, d_model)
        """
        residual = x
        x = self.layer_norm(x)
        x = self.attention(x, pos_emb, mask)
        x = self.dropout(x)
        return residual + x


class ConvolutionModule(nn.Module):
    """
    Convolution Module with GLU activation and depthwise convolution.
    """
    def __init__(self, d_model, kernel_size=32, expansion_factor=2, dropout=0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Pointwise expansion with GLU
        self.pointwise_conv1 = nn.Conv1d(
            d_model, 
            d_model * expansion_factor, 
            kernel_size=1
        )
        
        # GLU activation (splits channels in half)
        self.glu = nn.GLU(dim=1)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model
        )
        
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()  # Swish activation
        
        # Pointwise compression
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, time, d_model)
        Returns:
            (batch, time, d_model)
        """
        residual = x
        x = self.layer_norm(x)
        
        # Change to (batch, d_model, time)
        x = x.transpose(1, 2)
        
        # Pointwise expansion + GLU
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # Pointwise compression
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # Back to (batch, time, d_model)
        x = x.transpose(1, 2)
        
        return residual + x


class FeedForwardModule(nn.Module):
    """
    Feed-Forward Module with Swish activation and pre-norm.
    """
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.activation = nn.SiLU()  # Swish activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, time, d_model)
        Returns:
            (batch, time, d_model)
        """
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + x


class RelativePositionalEncoding(nn.Module):
    """
    Relative sinusoidal positional encoding.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.pe = self._generate_positional_encoding(max_len, d_model)
    
    def _generate_positional_encoding(self, max_len, d_model):
        """Generate sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x):
        """
        Args:
            x: (batch, time, d_model)
        Returns:
            (batch, time, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Get positional encoding
        pe = self.pe[:seq_len, :].to(x.device)
        
        # Expand to batch
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
        
        return pe