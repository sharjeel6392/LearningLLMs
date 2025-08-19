# Build a Large Language Model (From Scratch)
This repository documents my hands-on journey into the fascinating world of **Large Language Models (LLMs)**, following the detailed guidance provided in the book, **"Build a Large Language Model (From Scratch)"** by **Sebastian Raschka**. My goal is to demystify the core components of LLMs by implementing foundational concepts step-by-step, from data preparation to model architecture. Each section of this repository corresponds to a critical phase in the LLM development lifecycle, providing a practical, code-first approach to learning.

### Tokenization:
This section dives deep into the crucial first step of preparing textual data for a model. Here, I explore the fundamental concepts behind tokenization, the process of breaking down raw text into smaller, manageable units called tokens. I will demonstrate various tokenization strategies, including:

* **Word-based Tokenization:** A simple method where each word is a token.

* **Character-based Tokenization:** Breaking down text into individual characters.

* **Subword Tokenization (e.g., Byte-Pair Encoding):** A more advanced and widely used technique that balances vocabulary size and token granularity by creating tokens from frequently occurring character sequences.

The code in this section is designed to be highly illustrative, showing how to transform a human-readable corpus into a sequence of numerical token IDs that an LLM can process.

### Simple Self Attention:
This section focuses on the self-attention mechanism, a cornerstone of the Transformer architecture that has revolutionized natural language processing. Unlike traditional recurrent neural networks, self-attention allows a model to weigh the importance of all other words in a sequence when processing a single word, regardless of their position.

In this dedicated module, I implement a simplified version of the self-attention mechanism. The design is intentionally streamlined to facilitate a clear understanding of its internal workings:

* **No Trainable Parameters:** To isolate the core logic, this implementation has no learnable weights. This allows for a focus on the matrix multiplication and scoring process without the complexity of backpropagation.

* **Forged Inputs:** The inputs are simplified with low-dimensional embeddings (e.g., 3 dimensions) to make it possible to trace the flow of data and verify calculations manually.

This section provides a clear, step-by-step walkthrough of how attention scores are calculated, how they are normalized using a softmax function, and how they are used to compute a context-aware representation for each token in the sequence. This foundational understanding is essential before moving on to the more complex, full-scale Transformer model.