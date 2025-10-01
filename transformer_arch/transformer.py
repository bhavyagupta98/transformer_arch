"""
Complete transformer architecture implementation.

This module implements the full transformer model as described in
"Attention Is All You Need" with both encoder-decoder and decoder-only variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy

from .attention import MultiHeadAttention
from .layers import EncoderLayer, DecoderLayer, LayerNorm
from .positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """
    Complete transformer model with encoder-decoder architecture.
    
    This implements the transformer model from "Attention Is All You Need"
    with both encoder and decoder stacks.
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_length=5000, dropout=0.1, activation="relu"):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_length, dropout)
        
        # Encoder and decoder stacks
        self.encoder = EncoderStack(d_model, num_heads, num_encoder_layers, d_ff, dropout)
        self.decoder = DecoderStack(d_model, num_heads, num_decoder_layers, d_ff, dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self.init_parameters()
    
    def init_parameters(self):
        """Initialize model parameters using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass of the transformer.
        
        Args:
            src: Source sequence tensor (batch_size, src_len)
            tgt: Target sequence tensor (batch_size, tgt_len)
            src_mask: Source sequence mask
            tgt_mask: Target sequence mask
            
        Returns:
            output: Transformer output (batch_size, tgt_len, tgt_vocab_size)
            attention_weights: Attention weights for visualization
        """
        # Embed and add positional encoding
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        src_embedded = self.positional_encoding(src_embedded.transpose(0, 1)).transpose(0, 1)
        tgt_embedded = self.positional_encoding(tgt_embedded.transpose(0, 1)).transpose(0, 1)
        
        # Pass through encoder and decoder
        encoder_output, encoder_attention = self.encoder(src_embedded, src_mask)
        decoder_output, decoder_attention = self.decoder(
            tgt_embedded, encoder_output, src_mask, tgt_mask
        )
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        
        return output, (encoder_attention, decoder_attention)


class EncoderStack(nn.Module):
    """
    Stack of encoder layers.
    """
    
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout):
        super(EncoderStack, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        Forward pass through encoder stack.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask
            
        Returns:
            output: Encoder output
            attention_weights: All attention weights
        """
        attention_weights = []
        
        for layer in self.layers:
            x, attention = layer(x, mask)
            attention_weights.append(attention)
        
        return self.norm(x), attention_weights


class DecoderStack(nn.Module):
    """
    Stack of decoder layers.
    """
    
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout):
        super(DecoderStack, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Forward pass through decoder stack.
        
        Args:
            x: Input tensor (batch_size, tgt_len, d_model)
            encoder_output: Encoder output
            src_mask: Source attention mask
            tgt_mask: Target attention mask
            
        Returns:
            output: Decoder output
            attention_weights: All attention weights
        """
        self_attention_weights = []
        cross_attention_weights = []
        
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            self_attention_weights.append(self_attn)
            cross_attention_weights.append(cross_attn)
        
        return self.norm(x), (self_attention_weights, cross_attention_weights)


class GPTDecoder(nn.Module):
    """
    Decoder-only transformer model (GPT-style).
    
    This implements a decoder-only transformer suitable for autoregressive
    language modeling tasks.
    """
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, max_length=5000, dropout=0.1):
        super(GPTDecoder, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length, dropout)
        
        # Decoder layers (without cross-attention)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.norm = LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self.init_parameters()
    
    def init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        """
        Forward pass of GPT decoder.
        
        Args:
            x: Input token ids (batch_size, seq_len)
            mask: Attention mask
            
        Returns:
            output: Logits over vocabulary (batch_size, seq_len, vocab_size)
            attention_weights: Attention weights
        """
        # Embed and add positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Pass through decoder layers
        attention_weights = []
        for layer in self.layers:
            # For GPT, we only use self-attention (no cross-attention)
            x, self_attn, _ = layer(x, None, None, mask)
            attention_weights.append(self_attn)
        
        # Final layer norm and projection
        x = self.norm(x)
        output = self.output_projection(x)
        
        return output, attention_weights
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=None, top_p=None):
        """
        Generate text using the GPT decoder.
        
        Args:
            input_ids: Starting token ids (batch_size, seq_len)
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            generated_ids: Generated token ids
        """
        self.eval()
        with torch.no_grad():
            generated = input_ids.clone()
            
            for _ in range(max_length - input_ids.size(1)):
                # Get predictions for the last token
                logits, _ = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
