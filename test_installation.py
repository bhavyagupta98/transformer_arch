#!/usr/bin/env python3
"""
Test script to verify the transformer architecture installation and basic functionality.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from transformer_arch.transformer import Transformer, GPTDecoder
        print("✓ Transformer modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import transformer modules: {e}")
        return False
    
    try:
        from transformer_arch.attention import MultiHeadAttention, ScaledDotProductAttention
        print("✓ Attention modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import attention modules: {e}")
        return False
    
    try:
        from transformer_arch.positional_encoding import PositionalEncoding
        print("✓ Positional encoding modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import positional encoding modules: {e}")
        return False
    
    try:
        from transformer_arch.utils import get_device, create_padding_mask
        print("✓ Utility modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import utility modules: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality of the transformer components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test device detection
        from transformer_arch.utils import get_device
        device = get_device()
        print(f"✓ Device detection: {device}")
        
        # Test attention mechanism
        from transformer_arch.attention import MultiHeadAttention
        attention = MultiHeadAttention(d_model=64, num_heads=4)
        x = torch.randn(1, 8, 64)
        output, weights = attention(x, x, x)
        print(f"✓ Attention mechanism: input {x.shape} -> output {output.shape}")
        
        # Test positional encoding
        from transformer_arch.positional_encoding import PositionalEncoding
        pos_encoding = PositionalEncoding(d_model=64, max_length=100)
        x = torch.randn(1, 10, 64)
        encoded = pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        print(f"✓ Positional encoding: input {x.shape} -> output {encoded.shape}")
        
        # Test GPT decoder
        from transformer_arch.transformer import GPTDecoder
        gpt = GPTDecoder(vocab_size=100, d_model=64, num_heads=4, num_layers=2, d_ff=256)
        input_ids = torch.randint(0, 100, (1, 10))
        output, attention = gpt(input_ids)
        print(f"✓ GPT decoder: input {input_ids.shape} -> output {output.shape}")
        
        # Test encoder-decoder transformer
        from transformer_arch.transformer import Transformer
        transformer = Transformer(
            src_vocab_size=100, 
            tgt_vocab_size=100, 
            d_model=64, 
            num_heads=4, 
            num_encoder_layers=2, 
            num_decoder_layers=2, 
            d_ff=256
        )
        src = torch.randint(0, 100, (1, 8))
        tgt = torch.randint(0, 100, (1, 6))
        output, attention = transformer(src, tgt)
        print(f"✓ Encoder-decoder transformer: src {src.shape}, tgt {tgt.shape} -> output {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_examples():
    """Test that example scripts can be imported."""
    print("\nTesting example imports...")
    
    try:
        # Test text generation example
        sys.path.append(os.path.join(os.path.dirname(__file__), 'examples'))
        from text_generation import SimpleTokenizer
        tokenizer = SimpleTokenizer("hello world")
        print("✓ Text generation example imports successfully")
        
        # Test translation example
        from translation import TranslationDataset
        print("✓ Translation example imports successfully")
        
        # Test attention visualization example
        from attention_visualization import create_sample_model_and_data
        print("✓ Attention visualization example imports successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Example imports failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Transformer Architecture Installation")
    print("=" * 50)
    
    # Test PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_basic_functionality():
        tests_passed += 1
    
    if test_examples():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The installation is working correctly.")
        print("\nYou can now:")
        print("1. Run 'python demo.py' for a quick demonstration")
        print("2. Explore the examples/ directory")
        print("3. Check out the notebooks/ directory")
        print("4. Read the README.md for detailed documentation")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
