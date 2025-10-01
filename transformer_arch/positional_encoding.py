"""
Positional encoding for transformer architecture.

This module implements various positional encoding schemes to provide
position information to the transformer model since it has no inherent
notion of sequence order.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in "Attention Is All You Need".
    
    This encoding adds position information to input embeddings using
    sine and cosine functions of different frequencies.
    """
    
    def __init__(self, d_model, max_length=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        # Calculate div_term for the sinusoidal functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            output: Input with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding where position embeddings are learned parameters.
    
    This is an alternative to sinusoidal encoding where the model learns
    the optimal positional representations during training.
    """
    
    def __init__(self, d_model, max_length=5000, dropout=0.1):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(max_length, d_model)
        
    def forward(self, x):
        """
        Add learned positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            output: Input with learned positional encoding added
        """
        seq_len = x.size(0)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1).expand(seq_len, x.size(1))
        pos_encoding = self.embedding(positions).transpose(0, 1)
        x = x + pos_encoding
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding that considers relative positions between tokens.
    
    This encoding is used in some modern transformer variants and can be more
    effective for certain tasks than absolute positional encoding.
    """
    
    def __init__(self, d_model, max_relative_position=16, dropout=0.1):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        self.dropout = nn.Dropout(dropout)
        
        # Learnable embeddings for relative positions
        self.relative_position_embeddings = nn.Embedding(
            2 * max_relative_position + 1, d_model
        )
        
    def forward(self, x):
        """
        Add relative positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            output: Input with relative positional encoding added
        """
        batch_size, seq_len, d_model = x.size()
        
        # Create relative position indices
        range_vec = torch.arange(seq_len, device=x.device)
        range_mat = range_vec.unsqueeze(0).expand(seq_len, -1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip distances to max_relative_position
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        
        # Shift to positive indices
        final_mat = distance_mat_clipped + self.max_relative_position
        
        # Get relative position embeddings
        relative_position_embeddings = self.relative_position_embeddings(final_mat)
        
        # Add to input (simplified version - in practice, this would be more complex)
        x = x + relative_position_embeddings.mean(dim=0).unsqueeze(0)
        
        return self.dropout(x)


def create_positional_encoding(encoding_type="sinusoidal", d_model=512, max_length=5000, dropout=0.1):
    """
    Factory function to create different types of positional encodings.
    
    Args:
        encoding_type: Type of encoding ("sinusoidal", "learned", "relative")
        d_model: Model dimension
        max_length: Maximum sequence length
        dropout: Dropout rate
        
    Returns:
        Positional encoding module
    """
    if encoding_type == "sinusoidal":
        return PositionalEncoding(d_model, max_length, dropout)
    elif encoding_type == "learned":
        return LearnedPositionalEncoding(d_model, max_length, dropout)
    elif encoding_type == "relative":
        return RelativePositionalEncoding(d_model, dropout=dropout)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
