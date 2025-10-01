#!/usr/bin/env python3
"""
Transformer Architecture Demo Script

This script provides a quick demonstration of the transformer architecture
implementation with various examples and visualizations.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformer_arch.transformer import GPTDecoder, Transformer
from transformer_arch.attention import MultiHeadAttention
from transformer_arch.positional_encoding import PositionalEncoding
from transformer_arch.utils import get_device


def demo_attention_mechanism():
    """Demonstrate the attention mechanism."""
    print("=" * 50)
    print("ATTENTION MECHANISM DEMO")
    print("=" * 50)
    
    # Create attention layer
    d_model = 64
    num_heads = 4
    attention = MultiHeadAttention(d_model, num_heads)
    
    # Sample input
    batch_size, seq_len = 1, 8
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    print(f"Model dimension: {d_model}")
    print(f"Number of heads: {num_heads}")
    
    # Forward pass
    output, attn_weights = attention(x, x, x)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Visualize attention
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_weights[0, 0].detach().numpy(), cmap='Blues')
    plt.title('Attention Weights - Head 0')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.colorbar()
    plt.show()
    
    print("✓ Attention mechanism demo completed!")


def demo_positional_encoding():
    """Demonstrate positional encoding."""
    print("\n" + "=" * 50)
    print("POSITIONAL ENCODING DEMO")
    print("=" * 50)
    
    # Create positional encoding
    d_model = 64
    max_length = 50
    pos_encoding = PositionalEncoding(d_model, max_length)
    
    print(f"Model dimension: {d_model}")
    print(f"Max length: {max_length}")
    
    # Visualize positional encoding
    plt.figure(figsize=(12, 8))
    pos_enc_matrix = pos_encoding.pe[:max_length, 0, :].numpy()
    plt.imshow(pos_enc_matrix.T, cmap='RdBu', aspect='auto')
    plt.title('Positional Encoding Pattern')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.colorbar()
    plt.show()
    
    print("✓ Positional encoding demo completed!")


def demo_text_generation():
    """Demonstrate text generation with GPT."""
    print("\n" + "=" * 50)
    print("TEXT GENERATION DEMO")
    print("=" * 50)
    
    # Simple tokenizer
    class SimpleTokenizer:
        def __init__(self, text):
            self.chars = sorted(list(set(text)))
            self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
            self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
            self.vocab_size = len(self.chars)
        
        def encode(self, text):
            return [self.char_to_idx.get(char, 0) for char in text]
        
        def decode(self, tokens):
            return ''.join([self.idx_to_char.get(idx, '') for idx in tokens])
    
    # Sample text
    text = "The quick brown fox jumps over the lazy dog."
    tokenizer = SimpleTokenizer(text)
    
    print(f"Sample text: {text}")
    print(f"Vocabulary: {tokenizer.chars}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create GPT model
    device = get_device()
    gpt = GPTDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_length=50,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in gpt.parameters()):,}")
    
    # Simple training
    print("Training model...")
    gpt.train()
    optimizer = torch.optim.Adam(gpt.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(device)
    target_ids = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0).to(device)
    
    losses = []
    for epoch in range(10):
        optimizer.zero_grad()
        logits, _ = gpt(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # Generate text
    print("Generating text...")
    gpt.eval()
    
    def generate_text(prompt, max_length=20):
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
        with torch.no_grad():
            generated = gpt.generate(input_ids, max_length=max_length, temperature=0.8)
        return tokenizer.decode(generated[0].cpu().numpy())
    
    prompts = ["The", "A", "I"]
    for prompt in prompts:
        generated = generate_text(prompt, max_length=15)
        print(f"'{prompt}' -> '{generated}'")
    
    print("✓ Text generation demo completed!")


def demo_translation():
    """Demonstrate machine translation."""
    print("\n" + "=" * 50)
    print("MACHINE TRANSLATION DEMO")
    print("=" * 50)
    
    # Sample data
    source_texts = ["hello world", "how are you", "good morning"]
    target_texts = ["bonjour le monde", "comment allez vous", "bonjour"]
    
    print("Sample translation pairs:")
    for src, tgt in zip(source_texts, target_texts):
        print(f"  {src} -> {tgt}")
    
    # Create tokenizers
    class SimpleTokenizer:
        def __init__(self, texts):
            all_words = []
            for text in texts:
                words = text.lower().split()
                all_words.extend(words)
            
            unique_words = sorted(list(set(all_words)))
            self.word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
            self.idx_to_word = {idx: word for idx, word in enumerate(unique_words)}
            self.vocab_size = len(unique_words)
        
        def encode(self, text):
            words = text.lower().split()
            return [self.word_to_idx.get(word, 0) for word in words]
        
        def decode(self, tokens):
            words = [self.idx_to_word.get(idx, '<UNK>') for idx in tokens]
            return ' '.join(words)
    
    src_tokenizer = SimpleTokenizer(source_texts)
    tgt_tokenizer = SimpleTokenizer(target_texts)
    
    print(f"Source vocabulary size: {src_tokenizer.vocab_size}")
    print(f"Target vocabulary size: {tgt_tokenizer.vocab_size}")
    
    # Create translation model
    device = get_device()
    transformer = Transformer(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        d_model=32,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    # Test forward pass
    src = torch.randint(0, src_tokenizer.vocab_size, (1, 5))
    tgt = torch.randint(0, tgt_tokenizer.vocab_size, (1, 5))
    
    output, attention = transformer(src, tgt)
    print(f"Output shape: {output.shape}")
    
    print("✓ Machine translation demo completed!")


def demo_model_comparison():
    """Compare different model sizes."""
    print("\n" + "=" * 50)
    print("MODEL COMPARISON DEMO")
    print("=" * 50)
    
    # Model configurations
    configs = {
        'Small': {'d_model': 64, 'num_heads': 4, 'num_layers': 2, 'd_ff': 256},
        'Medium': {'d_model': 128, 'num_heads': 8, 'num_layers': 4, 'd_ff': 512},
        'Large': {'d_model': 256, 'num_heads': 8, 'num_layers': 6, 'd_ff': 1024}
    }
    
    results = {}
    for name, config in configs.items():
        model = GPTDecoder(vocab_size=1000, **config, dropout=0.1)
        param_count = sum(p.numel() for p in model.parameters())
        results[name] = param_count
        print(f"{name} model: {param_count:,} parameters")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    params = list(results.values())
    
    plt.bar(names, params)
    plt.title('Model Size vs Parameter Count')
    plt.xlabel('Model Size')
    plt.ylabel('Number of Parameters')
    plt.yscale('log')
    plt.show()
    
    print("✓ Model comparison demo completed!")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Transformer Architecture Demo')
    parser.add_argument('--demo', choices=['all', 'attention', 'positional', 'text', 'translation', 'comparison'],
                       default='all', help='Which demo to run')
    parser.add_argument('--no-plots', action='store_true', help='Disable plotting')
    
    args = parser.parse_args()
    
    if args.no_plots:
        plt.ioff()
    
    print("🚀 Transformer Architecture Demo")
    print("=" * 50)
    print(f"Device: {get_device()}")
    print(f"PyTorch version: {torch.__version__}")
    
    if args.demo in ['all', 'attention']:
        demo_attention_mechanism()
    
    if args.demo in ['all', 'positional']:
        demo_positional_encoding()
    
    if args.demo in ['all', 'text']:
        demo_text_generation()
    
    if args.demo in ['all', 'translation']:
        demo_translation()
    
    if args.demo in ['all', 'comparison']:
        demo_model_comparison()
    
    print("\n" + "=" * 50)
    print("🎉 All demos completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Explore the examples/ directory for more detailed examples")
    print("2. Check out the notebooks/ directory for interactive tutorials")
    print("3. Read the README.md for comprehensive documentation")
    print("4. Experiment with different model configurations")


if __name__ == "__main__":
    main()
