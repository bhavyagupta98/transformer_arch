"""
Utility functions for transformer architecture.

This module provides helper functions for creating masks, data preprocessing,
and other common operations used in transformer models.
"""

import torch
import torch.nn as nn
import numpy as np


def create_padding_mask(seq, pad_token=0):
    """
    Create padding mask to ignore padding tokens in attention.
    
    Args:
        seq: Input sequence tensor (batch_size, seq_len)
        pad_token: Padding token id
        
    Returns:
        mask: Padding mask (batch_size, 1, 1, seq_len)
    """
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(seq_len):
    """
    Create look-ahead mask to prevent attention to future tokens.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        mask: Look-ahead mask (seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0


def create_combined_mask(tgt, pad_token=0):
    """
    Create combined mask for target sequence (padding + look-ahead).
    
    Args:
        tgt: Target sequence tensor (batch_size, tgt_len)
        pad_token: Padding token id
        
    Returns:
        mask: Combined mask (batch_size, 1, tgt_len, tgt_len)
    """
    tgt_len = tgt.size(1)
    
    # Create padding mask
    padding_mask = create_padding_mask(tgt, pad_token)
    
    # Create look-ahead mask
    look_ahead_mask = create_look_ahead_mask(tgt_len)
    
    # Combine masks
    combined_mask = padding_mask & look_ahead_mask
    
    return combined_mask


def get_angles(pos, i, d_model):
    """
    Get angles for positional encoding.
    
    Args:
        pos: Position indices
        i: Dimension indices
        d_model: Model dimension
        
    Returns:
        angles: Angles for sinusoidal encoding
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding_1d(position, d_model):
    """
    Create 1D positional encoding.
    
    Args:
        position: Position tensor
        d_model: Model dimension
        
    Returns:
        pos_encoding: Positional encoding tensor
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis, :],
                           d_model)
    
    # Apply sin to even indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # Apply cos to odd indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        total_params: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model):
    """
    Initialize model weights using Xavier uniform initialization.
    
    Args:
        model: PyTorch model
    """
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def create_src_mask(src, pad_token=0):
    """
    Create source mask for encoder.
    
    Args:
        src: Source sequence tensor (batch_size, src_len)
        pad_token: Padding token id
        
    Returns:
        mask: Source mask (batch_size, 1, 1, src_len)
    """
    return create_padding_mask(src, pad_token)


def create_tgt_mask(tgt, pad_token=0):
    """
    Create target mask for decoder.
    
    Args:
        tgt: Target sequence tensor (batch_size, tgt_len)
        pad_token: Padding token id
        
    Returns:
        mask: Target mask (batch_size, 1, tgt_len, tgt_len)
    """
    return create_combined_mask(tgt, pad_token)


def get_device():
    """
    Get the best available device (CUDA if available, otherwise CPU).
    
    Returns:
        device: PyTorch device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model
        optimizer: Optional optimizer
        
    Returns:
        epoch: Epoch number
        loss: Loss value
    """
    checkpoint = torch.load(filepath, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def calculate_bleu_score(predictions, targets, n_gram=4):
    """
    Calculate BLEU score for evaluation.
    
    Args:
        predictions: List of predicted sequences
        targets: List of target sequences
        n_gram: Maximum n-gram to consider
        
    Returns:
        bleu_score: BLEU score
    """
    from collections import Counter
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def precision(pred_ngrams, target_ngrams, n):
        if len(pred_ngrams) == 0:
            return 0
        counts = Counter(pred_ngrams)
        clipped_counts = Counter(target_ngrams)
        clipped_counts = {k: min(counts[k], clipped_counts[k]) for k in counts}
        return sum(clipped_counts.values()) / len(pred_ngrams)
    
    precisions = []
    for n in range(1, n_gram + 1):
        pred_ngrams = []
        target_ngrams = []
        
        for pred, target in zip(predictions, targets):
            pred_ngrams.extend(get_ngrams(pred, n))
            target_ngrams.extend(get_ngrams(target, n))
        
        precisions.append(precision(pred_ngrams, target_ngrams, n))
    
    # Calculate geometric mean of precisions
    if min(precisions) > 0:
        bleu = np.exp(np.mean(np.log(precisions)))
    else:
        bleu = 0
    
    return bleu


def create_learning_rate_scheduler(optimizer, d_model, warmup_steps=4000):
    """
    Create learning rate scheduler with warmup.
    
    Args:
        optimizer: PyTorch optimizer
        d_model: Model dimension
        warmup_steps: Number of warmup steps
        
    Returns:
        scheduler: Learning rate scheduler
    """
    def lr_lambda(step):
        if step == 0:
            step = 1
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
