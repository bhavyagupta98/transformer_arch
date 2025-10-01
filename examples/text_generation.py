"""
Text Generation Example using GPT-style Decoder

This example demonstrates how to use the transformer architecture for
autoregressive text generation tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

# Add parent directory to path to import transformer_arch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_arch.transformer import GPTDecoder
from transformer_arch.utils import get_device, create_look_ahead_mask


class TextDataset(Dataset):
    """Dataset for text generation training."""
    
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens.extend([0] * (self.max_length - len(tokens)))
        
        # Input is all tokens except the last one
        # Target is all tokens except the first one
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids


class SimpleTokenizer:
    """Simple character-level tokenizer for demonstration."""
    
    def __init__(self, text):
        # Create vocabulary from unique characters
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    
    def encode(self, text):
        """Convert text to token indices."""
        return [self.char_to_idx.get(char, 0) for char in text]
    
    def decode(self, tokens):
        """Convert token indices to text."""
        return ''.join([self.idx_to_char.get(idx, '') for idx in tokens])


def train_model(model, dataloader, optimizer, criterion, device, num_epochs=10):
    """Train the text generation model."""
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(input_ids)
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
    
    return losses


def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, device=None):
    """Generate text using the trained model."""
    model.eval()
    
    if device is None:
        device = get_device()
    
    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        generated = model.generate(
            input_ids, 
            max_length=max_length, 
            temperature=temperature
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(generated[0].cpu().numpy())
    return generated_text


def main():
    """Main function to demonstrate text generation."""
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Sample text for training
    sample_text = """
    The quick brown fox jumps over the lazy dog. This is a sample text for training
    a transformer model for text generation. The model will learn to predict the next
    character based on the previous characters. This is a simple example but it
    demonstrates the core concepts of autoregressive language modeling.
    """
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(sample_text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Characters: {tokenizer.chars}")
    
    # Create dataset
    texts = [sample_text.strip()]
    dataset = TextDataset(texts, tokenizer, max_length=64)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Create model
    model = GPTDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=3,
        d_ff=512,
        max_length=128,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    print("Training model...")
    losses = train_model(model, dataloader, optimizer, criterion, device, num_epochs=20)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # Generate text
    print("\nGenerating text...")
    prompts = ["The", "This", "A"]
    
    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, max_length=50, temperature=0.8)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'\n")


if __name__ == "__main__":
    main()
