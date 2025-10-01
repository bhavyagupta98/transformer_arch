"""
Core building blocks for transformer architecture.

This module implements the fundamental layers used in transformer models
including feed-forward networks, layer normalization, and residual connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    This implements the feed-forward network used in each transformer layer:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1, activation="relu"):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "swish":
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        """
        Forward pass of the feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            output: Feed-forward output
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class LayerNorm(nn.Module):
    """
    Layer normalization with learnable parameters.
    
    Layer normalization normalizes the inputs across the features dimension
    for each sample independently.
    """
    
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        """
        Forward pass of layer normalization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            output: Layer normalized output
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualConnection(nn.Module):
    """
    Residual connection with layer normalization.
    
    This implements the residual connection pattern used throughout
    the transformer architecture: output = LayerNorm(x + Sublayer(x))
    """
    
    def __init__(self, d_model, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer.
        
        Args:
            x: Input tensor
            sublayer: Function to apply to x
            
        Returns:
            output: Residual connection output
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    Single encoder layer with self-attention and feed-forward network.
    
    Each encoder layer consists of:
    1. Multi-head self-attention with residual connection
    2. Position-wise feed-forward network with residual connection
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(2)
        ])
    
    def forward(self, x, mask=None):
        """
        Forward pass of encoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Encoder layer output
            attention_weights: Self-attention weights
        """
        # Self-attention with residual connection
        x, attention_weights = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, mask)
        )
        
        # Feed-forward with residual connection
        x = self.residual_connections[1](x, self.feed_forward)
        
        return x, attention_weights


class DecoderLayer(nn.Module):
    """
    Single decoder layer with masked self-attention, cross-attention, and feed-forward.
    
    Each decoder layer consists of:
    1. Masked multi-head self-attention with residual connection
    2. Multi-head cross-attention with residual connection
    3. Position-wise feed-forward network with residual connection
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(3)
        ])
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Forward pass of decoder layer.
        
        Args:
            x: Input tensor from previous decoder layer
            encoder_output: Output from encoder
            src_mask: Source sequence mask
            tgt_mask: Target sequence mask
            
        Returns:
            output: Decoder layer output
            self_attention_weights: Self-attention weights
            cross_attention_weights: Cross-attention weights
        """
        # Masked self-attention with residual connection
        x, self_attention_weights = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, tgt_mask)
        )
        
        # Cross-attention with residual connection
        x, cross_attention_weights = self.residual_connections[1](
            x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask)
        )
        
        # Feed-forward with residual connection
        x = self.residual_connections[2](x, self.feed_forward)
        
        return x, self_attention_weights, cross_attention_weights


# Import MultiHeadAttention here to avoid circular imports
from .attention import MultiHeadAttention
