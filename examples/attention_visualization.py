"""
Attention Visualization Example

This example demonstrates how to visualize attention weights in transformer models
to understand what the model is focusing on during processing.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# Add parent directory to path to import transformer_arch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_arch.transformer import GPTDecoder
from transformer_arch.utils import get_device


def visualize_attention(attention_weights, tokens, layer_idx=0, head_idx=0, title="Attention Weights"):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weights tensor
        tokens: List of tokens
        layer_idx: Which layer to visualize
        head_idx: Which attention head to visualize
        title: Plot title
    """
    # Extract attention weights for specific layer and head
    if isinstance(attention_weights, list):
        attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()  # [seq_len, seq_len]
    else:
        attn = attention_weights[0, head_idx].cpu().numpy()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, 
                xticklabels=tokens, 
                yticklabels=tokens,
                cmap='Blues',
                cbar=True,
                square=True)
    plt.title(f"{title} - Layer {layer_idx}, Head {head_idx}")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def visualize_multi_head_attention(attention_weights, tokens, layer_idx=0, title="Multi-Head Attention"):
    """
    Visualize all attention heads for a given layer.
    
    Args:
        attention_weights: Attention weights tensor
        tokens: List of tokens
        layer_idx: Which layer to visualize
        title: Plot title
    """
    if isinstance(attention_weights, list):
        attn = attention_weights[layer_idx][0]  # [num_heads, seq_len, seq_len]
    else:
        attn = attention_weights[0]  # [num_heads, seq_len, seq_len]
    
    num_heads = attn.size(0)
    fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=(15, 10))
    axes = axes.flatten() if num_heads > 1 else [axes]
    
    for head_idx in range(num_heads):
        attn_matrix = attn[head_idx].cpu().numpy()
        
        sns.heatmap(attn_matrix,
                   xticklabels=tokens,
                   yticklabels=tokens,
                   cmap='Blues',
                   cbar=True,
                   square=True,
                   ax=axes[head_idx])
        axes[head_idx].set_title(f"Head {head_idx}")
        axes[head_idx].set_xlabel("Key Position")
        axes[head_idx].set_ylabel("Query Position")
    
    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f"{title} - Layer {layer_idx}")
    plt.tight_layout()
    plt.show()


def analyze_attention_patterns(attention_weights, tokens, layer_idx=0):
    """
    Analyze attention patterns to understand model behavior.
    
    Args:
        attention_weights: Attention weights tensor
        tokens: List of tokens
        layer_idx: Which layer to analyze
    """
    if isinstance(attention_weights, list):
        attn = attention_weights[layer_idx][0]  # [num_heads, seq_len, seq_len]
    else:
        attn = attention_weights[0]  # [num_heads, seq_len, seq_len]
    
    num_heads, seq_len, _ = attn.shape
    
    print(f"\\n=== Attention Analysis for Layer {layer_idx} ===")
    print(f"Sequence length: {seq_len}")
    print(f"Number of heads: {num_heads}")
    
    # Analyze each head
    for head_idx in range(num_heads):
        head_attn = attn[head_idx].cpu().numpy()
        
        # Find most attended positions for each query position
        max_attentions = np.argmax(head_attn, axis=1)
        
        print(f"\\nHead {head_idx}:")
        for i, (token, max_pos) in enumerate(zip(tokens, max_attentions)):
            max_attn_score = head_attn[i, max_pos]
            attended_token = tokens[max_pos]
            print(f"  '{token}' (pos {i}) -> '{attended_token}' (pos {max_pos}) [score: {max_attn_score:.3f}]")
        
        # Calculate attention entropy (diversity of attention)
        entropy = -np.sum(head_attn * np.log(head_attn + 1e-9), axis=1)
        avg_entropy = np.mean(entropy)
        print(f"  Average attention entropy: {avg_entropy:.3f}")


def create_sample_model_and_data():
    """Create a simple model and sample data for demonstration."""
    device = get_device()
    
    # Sample text
    text = "The quick brown fox jumps over the lazy dog"
    tokens = text.split()
    
    # Create simple tokenizer
    vocab = {token: idx for idx, token in enumerate(set(tokens))}
    vocab['<PAD>'] = len(vocab)
    vocab_size = len(vocab)
    
    # Convert text to tokens
    token_ids = [vocab[token] for token in tokens]
    input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    
    # Create model
    model = GPTDecoder(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_length=len(tokens),
        dropout=0.0  # No dropout for visualization
    ).to(device)
    
    return model, input_tensor, tokens


def main():
    """Main function to demonstrate attention visualization."""
    print("Creating sample model and data...")
    model, input_tensor, tokens = create_sample_model_and_data()
    
    print(f"Tokens: {tokens}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get attention weights
    model.eval()
    with torch.no_grad():
        _, attention_weights = model(input_tensor)
    
    print(f"\\nNumber of layers: {len(attention_weights)}")
    print(f"Attention weights shape: {attention_weights[0].shape}")
    
    # Visualize attention for different layers and heads
    for layer_idx in range(len(attention_weights)):
        print(f"\\n=== Layer {layer_idx} ===")
        
        # Visualize all heads
        visualize_multi_head_attention(
            attention_weights, 
            tokens, 
            layer_idx=layer_idx,
            title="Multi-Head Attention"
        )
        
        # Visualize individual heads
        num_heads = attention_weights[layer_idx].size(1)
        for head_idx in range(min(2, num_heads)):  # Show first 2 heads
            visualize_attention(
                attention_weights,
                tokens,
                layer_idx=layer_idx,
                head_idx=head_idx,
                title="Attention Weights"
            )
        
        # Analyze attention patterns
        analyze_attention_patterns(attention_weights, tokens, layer_idx=layer_idx)


def demonstrate_attention_mechanisms():
    """Demonstrate different attention mechanisms."""
    print("\\n" + "="*50)
    print("ATTENTION MECHANISM DEMONSTRATION")
    print("="*50)
    
    # Create sample data
    seq_len = 8
    d_model = 64
    num_heads = 4
    
    # Random input
    x = torch.randn(1, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    print(f"Sequence length: {seq_len}")
    print(f"Model dimension: {d_model}")
    print(f"Number of heads: {num_heads}")
    
    # Create attention layer
    from transformer_arch.attention import MultiHeadAttention
    attention = MultiHeadAttention(d_model, num_heads)
    
    # Forward pass
    output, attn_weights = attention(x, x, x)
    
    print(f"\\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Visualize attention weights
    tokens = [f"token_{i}" for i in range(seq_len)]
    visualize_attention(attn_weights, tokens, head_idx=0, title="Self-Attention")
    
    # Show attention statistics
    attn_matrix = attn_weights[0, 0].cpu().numpy()
    print(f"\\nAttention Statistics:")
    print(f"  Mean attention weight: {np.mean(attn_matrix):.4f}")
    print(f"  Max attention weight: {np.max(attn_matrix):.4f}")
    print(f"  Min attention weight: {np.min(attn_matrix):.4f}")
    print(f"  Attention sparsity: {np.sum(attn_matrix < 0.01) / attn_matrix.size:.4f}")


if __name__ == "__main__":
    main()
    demonstrate_attention_mechanisms()
