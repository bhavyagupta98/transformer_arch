# Transformer Architecture Demo

A comprehensive Python project demonstrating transformer architectures and their use cases. This project provides a complete implementation of the transformer model from "Attention Is All You Need" along with practical examples for text generation, machine translation, and attention visualization.

## 🚀 Features

- **Complete Transformer Implementation**: From scratch implementation of encoder-decoder and decoder-only transformers
- **Attention Mechanisms**: Multi-head attention, scaled dot-product attention, and self-attention
- **Positional Encoding**: Sinusoidal, learned, and relative positional encoding schemes
- **Practical Examples**: Text generation, machine translation, and attention visualization
- **Educational Notebooks**: Step-by-step tutorials and visualizations
- **Modular Design**: Clean, well-documented code that's easy to understand and extend

## 📁 Project Structure

```
transformer_arch/
├── transformer_arch/           # Core transformer implementation
│   ├── __init__.py
│   ├── attention.py            # Attention mechanisms
│   ├── layers.py               # Core building blocks
│   ├── positional_encoding.py  # Positional encoding schemes
│   ├── transformer.py          # Complete transformer models
│   └── utils.py                # Utility functions
├── examples/                   # Practical examples
│   ├── text_generation.py      # GPT-style text generation
│   ├── translation.py          # Machine translation
│   └── attention_visualization.py  # Attention analysis
├── notebooks/                  # Jupyter notebooks
│   ├── quick_start.ipynb       # Quick start guide
│   └── transformer_tutorial.ipynb  # Comprehensive tutorial
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd transformer_arch
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package (optional):**
   ```bash
   pip install -e .
   ```

## 🚀 Quick Start

### Basic Usage

```python
import torch
from transformer_arch.transformer import GPTDecoder
from transformer_arch.attention import MultiHeadAttention

# Create a GPT-style decoder
model = GPTDecoder(
    vocab_size=1000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    dropout=0.1
)

# Generate text
input_ids = torch.tensor([[1, 2, 3, 4]])  # Your input tokens
output, attention_weights = model(input_ids)
```

### Text Generation Example

```python
from examples.text_generation import main as run_text_generation

# Run the text generation example
run_text_generation()
```

### Machine Translation Example

```python
from examples.translation import main as run_translation

# Run the translation example
run_translation()
```

## 📚 Examples

### 1. Text Generation (GPT-style)

The text generation example demonstrates how to train a decoder-only transformer for autoregressive language modeling:

```bash
python examples/text_generation.py
```

**Features:**
- Character-level tokenization
- Autoregressive text generation
- Temperature and top-k sampling
- Training visualization

### 2. Machine Translation

The translation example shows how to use an encoder-decoder transformer for sequence-to-sequence tasks:

```bash
python examples/translation.py
```

**Features:**
- Word-level tokenization
- Encoder-decoder architecture
- Attention masking
- BLEU score evaluation

### 3. Attention Visualization

The attention visualization example helps understand what the model learns:

```bash
python examples/attention_visualization.py
```

**Features:**
- Attention weight heatmaps
- Multi-head attention analysis
- Attention pattern statistics
- Interactive visualizations

## 🧠 Architecture Overview

### Attention Mechanism

The core innovation of transformers is the attention mechanism:

```python
from transformer_arch.attention import MultiHeadAttention

# Multi-head attention
attention = MultiHeadAttention(d_model=512, num_heads=8)
output, weights = attention(query, key, value)
```

### Positional Encoding

Transformers need positional information since they process sequences in parallel:

```python
from transformer_arch.positional_encoding import PositionalEncoding

# Sinusoidal positional encoding
pos_encoding = PositionalEncoding(d_model=512, max_length=5000)
encoded = pos_encoding(input_embeddings)
```

### Complete Transformer

```python
from transformer_arch.transformer import Transformer

# Encoder-decoder transformer
model = Transformer(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048
)
```

## 📖 Jupyter Notebooks

### Quick Start Notebook
- Basic attention mechanism
- Simple text generation
- Machine translation setup

### Comprehensive Tutorial
- Deep dive into attention mechanisms
- Positional encoding analysis
- Model architecture exploration
- Advanced visualization techniques

## 🔧 Key Components

### Attention Mechanisms
- **Scaled Dot-Product Attention**: Core attention computation
- **Multi-Head Attention**: Parallel attention heads
- **Self-Attention**: Query, key, and value from same input
- **Cross-Attention**: Query from decoder, key/value from encoder

### Positional Encoding
- **Sinusoidal**: Fixed sinusoidal patterns
- **Learned**: Trainable position embeddings
- **Relative**: Relative position encoding

### Model Variants
- **Encoder-Decoder**: For sequence-to-sequence tasks
- **Decoder-Only**: For autoregressive generation (GPT-style)
- **Encoder-Only**: For classification and feature extraction

## 📊 Performance

The implementation is optimized for educational purposes while maintaining reasonable performance:

- **Memory Efficient**: Gradient checkpointing support
- **GPU Accelerated**: CUDA support when available
- **Modular Design**: Easy to experiment with different configurations
- **Well Documented**: Comprehensive docstrings and comments

## 🎯 Use Cases

### Natural Language Processing
- Text generation and completion
- Machine translation
- Text summarization
- Question answering
- Sentiment analysis

### Computer Vision
- Image captioning
- Visual question answering
- Object detection (with modifications)

### Other Applications
- Time series forecasting
- Music generation
- Code generation
- Protein sequence analysis

## 🧪 Experiments

Try these experiments to understand transformers better:

1. **Attention Analysis**: Visualize attention patterns for different inputs
2. **Model Scaling**: Compare performance with different model sizes
3. **Positional Encoding**: Test different encoding schemes
4. **Training Dynamics**: Observe how attention patterns change during training
5. **Generation Quality**: Experiment with different sampling strategies

## 📚 Educational Value

This project is designed for learning and understanding:

- **Clear Implementation**: Well-commented code that's easy to follow
- **Step-by-Step Examples**: Gradual complexity from basic to advanced
- **Visualizations**: Rich visualizations to understand model behavior
- **Modular Design**: Easy to modify and experiment with
- **Comprehensive Documentation**: Detailed explanations of each component

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original Transformer paper: "Attention Is All You Need" by Vaswani et al.
- PyTorch team for the excellent deep learning framework
- The open-source community for inspiration and feedback

## 📞 Support

If you have questions or need help:

1. Check the examples and notebooks
2. Read the documentation
3. Open an issue on GitHub
4. Join our community discussions

---

**Happy Learning! 🚀**

Transformers have revolutionized deep learning, and this project aims to make them accessible and understandable for everyone. Whether you're a beginner or an expert, there's something here for you to learn and explore.