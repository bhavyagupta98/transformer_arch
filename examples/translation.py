"""
Machine Translation Example using Encoder-Decoder Transformer

This example demonstrates how to use the transformer architecture for
sequence-to-sequence tasks like machine translation.
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

from transformer_arch.transformer import Transformer
from transformer_arch.utils import get_device, create_src_mask, create_tgt_mask


class TranslationDataset(Dataset):
    """Dataset for machine translation training."""
    
    def __init__(self, source_texts, target_texts, src_tokenizer, tgt_tokenizer, max_length=64):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, idx):
        src_text = self.source_texts[idx]
        tgt_text = self.target_texts[idx]
        
        # Tokenize
        src_tokens = self.src_tokenizer.encode(src_text)
        tgt_tokens = self.tgt_tokenizer.encode(tgt_text)
        
        # Add special tokens
        src_tokens = [self.src_tokenizer.sos_token] + src_tokens + [self.src_tokenizer.eos_token]
        tgt_tokens = [self.tgt_tokenizer.sos_token] + tgt_tokens + [self.tgt_tokenizer.eos_token]
        
        # Truncate or pad
        if len(src_tokens) > self.max_length:
            src_tokens = src_tokens[:self.max_length]
        else:
            src_tokens.extend([self.src_tokenizer.pad_token] * (self.max_length - len(src_tokens)))
        
        if len(tgt_tokens) > self.max_length:
            tgt_tokens = tgt_tokens[:self.max_length]
        else:
            tgt_tokens.extend([self.tgt_tokenizer.pad_token] * (self.max_length - len(tgt_tokens)))
        
        return (torch.tensor(src_tokens, dtype=torch.long),
                torch.tensor(tgt_tokens, dtype=torch.long))


class SimpleTokenizer:
    """Simple word-level tokenizer for demonstration."""
    
    def __init__(self, texts, special_tokens=None):
        if special_tokens is None:
            special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        
        self.special_tokens = special_tokens
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2
        self.unk_token = 3
        
        # Build vocabulary
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
        
        # Create vocabulary
        unique_words = sorted(list(set(all_words)))
        self.word_to_idx = {word: idx + len(special_tokens) for idx, word in enumerate(unique_words)}
        self.idx_to_word = {idx + len(special_tokens): word for idx, word in enumerate(unique_words)}
        
        # Add special tokens
        for i, token in enumerate(special_tokens):
            self.word_to_idx[token] = i
            self.idx_to_word[i] = token
        
        self.vocab_size = len(self.word_to_idx)
    
    def encode(self, text):
        """Convert text to token indices."""
        words = text.lower().split()
        return [self.word_to_idx.get(word, self.unk_token) for word in words]
    
    def decode(self, tokens):
        """Convert token indices to text."""
        words = [self.idx_to_word.get(idx, '<UNK>') for idx in tokens]
        return ' '.join(words)


def train_model(model, dataloader, optimizer, criterion, device, num_epochs=10):
    """Train the translation model."""
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Create masks
            src_mask = create_src_mask(src, pad_token=0)
            tgt_mask = create_tgt_mask(tgt, pad_token=0)
            
            # Prepare input and target
            tgt_input = tgt[:, :-1]  # Remove last token
            tgt_output = tgt[:, 1:]  # Remove first token
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(src, tgt_input, src_mask, tgt_mask)
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
    
    return losses


def translate(model, src_tokenizer, tgt_tokenizer, src_text, max_length=50, device=None):
    """Translate text using the trained model."""
    model.eval()
    
    if device is None:
        device = get_device()
    
    # Encode source
    src_tokens = src_tokenizer.encode(src_text)
    src_tokens = [src_tokenizer.sos_token] + src_tokens + [src_tokenizer.eos_token]
    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
    
    # Create source mask
    src_mask = create_src_mask(src_tensor, pad_token=0)
    
    # Start with SOS token
    tgt_tokens = [tgt_tokenizer.sos_token]
    
    with torch.no_grad():
        for _ in range(max_length):
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
            tgt_mask = create_tgt_mask(tgt_tensor, pad_token=0)
            
            # Get predictions
            logits, _ = model(src_tensor, tgt_tensor, src_mask, tgt_mask)
            
            # Get next token
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            
            # Stop if EOS token
            if next_token == tgt_tokenizer.eos_token:
                break
            
            tgt_tokens.append(next_token)
    
    # Decode translation
    translation = tgt_tokenizer.decode(tgt_tokens[1:])  # Remove SOS token
    return translation


def main():
    """Main function to demonstrate machine translation."""
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Sample translation data (English to French)
    source_texts = [
        "hello world",
        "how are you",
        "good morning",
        "thank you very much",
        "have a nice day",
        "see you later",
        "what is your name",
        "where are you from",
        "nice to meet you",
        "how old are you"
    ]
    
    target_texts = [
        "bonjour le monde",
        "comment allez vous",
        "bonjour",
        "merci beaucoup",
        "passez une bonne journee",
        "a bientot",
        "quel est votre nom",
        "d ou venez vous",
        "ravi de vous rencontrer",
        "quel age avez vous"
    ]
    
    # Create tokenizers
    src_tokenizer = SimpleTokenizer(source_texts)
    tgt_tokenizer = SimpleTokenizer(target_texts)
    
    print(f"Source vocabulary size: {src_tokenizer.vocab_size}")
    print(f"Target vocabulary size: {tgt_tokenizer.vocab_size}")
    
    # Create dataset
    dataset = TranslationDataset(source_texts, target_texts, src_tokenizer, tgt_tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Create model
    model = Transformer(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
        max_length=64,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    # Train model
    print("Training model...")
    losses = train_model(model, dataloader, optimizer, criterion, device, num_epochs=50)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # Test translation
    print("\nTesting translation...")
    test_sentences = ["hello world", "how are you", "good morning"]
    
    for src_text in test_sentences:
        translation = translate(model, src_tokenizer, tgt_tokenizer, src_text)
        print(f"Source: '{src_text}'")
        print(f"Translation: '{translation}'\n")


if __name__ == "__main__":
    main()
