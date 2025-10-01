"""
Attention mechanisms for transformer architecture.

This module implements the core attention mechanisms including scaled dot-product
attention and multi-head attention as described in "Attention Is All You Need".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    This implements the attention mechanism from the original transformer paper:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (batch_size, num_heads, seq_len, d_k)
            key: Key tensor of shape (batch_size, num_heads, seq_len, d_k)
            value: Value tensor of shape (batch_size, num_heads, seq_len, d_v)
            mask: Optional mask tensor
            
        Returns:
            output: Attention output tensor
            attention_weights: Attention weights for visualization
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    This implements the multi-head attention from the transformer paper,
    which allows the model to jointly attend to information from different
    representation subspaces at different positions.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear transformations for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            output: Multi-head attention output
            attention_weights: Attention weights for visualization
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear transformations and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.w_o(attention_output)
        
        return output, attention_weights


class SelfAttention(nn.Module):
    """
    Self-attention layer where query, key, and value are the same input.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass of self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            output: Self-attention output
            attention_weights: Attention weights
        """
        return self.multi_head_attention(x, x, x, mask)


class CrossAttention(nn.Module):
    """
    Cross-attention layer for encoder-decoder architectures.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
    def forward(self, query, key_value, mask=None):
        """
        Forward pass of cross-attention.
        
        Args:
            query: Query tensor from decoder
            key_value: Key-value tensor from encoder
            mask: Optional mask tensor
            
        Returns:
            output: Cross-attention output
            attention_weights: Attention weights
        """
        return self.multi_head_attention(query, key_value, key_value, mask)
