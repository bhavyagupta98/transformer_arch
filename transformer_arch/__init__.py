"""
Transformer Architecture Demo Package

This package provides a comprehensive implementation of transformer architectures
and demonstrates various use cases including text generation, translation, and
sequence-to-sequence tasks.
"""

from .transformer import Transformer
from .attention import MultiHeadAttention, ScaledDotProductAttention
from .positional_encoding import PositionalEncoding
from .layers import FeedForward, LayerNorm, ResidualConnection
from .utils import create_padding_mask, create_look_ahead_mask

__version__ = "1.0.0"
__all__ = [
    "Transformer",
    "MultiHeadAttention", 
    "ScaledDotProductAttention",
    "PositionalEncoding",
    "FeedForward",
    "LayerNorm",
    "ResidualConnection",
    "create_padding_mask",
    "create_look_ahead_mask"
]
