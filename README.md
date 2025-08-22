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

### Attention Mechanisms:
This directory contains implementations of various attention mechanisms, a key component in modern deep learning models, especially in the field of Natural Language Processing. The implementations range from foundational concepts to more complex, practical architectures.

* **Simple self attention**
This is a foundational implementation of self-attention that operates without any trainable parameters. Unlike the standard self-attention mechanism, it does not use separate weight matrices to create Query, Key, and Value tensors. Instead, attention scores are calculated directly from the dot product of the input token embeddings themselves. This version is designed to illustrate the core concept of how tokens in a sequence can "attend" to each other based on their inherent similarities, serving as an educational first step before introducing learned parameters.

* **Self attention Version 1 (nn.Parameter)**
This implementation introduces trainable parameters by using `nn.Parameter` to define the weight matrices for the **Query**, **Key**, and **Value** projections. `nn.Parameter` makes these tensors part of the model's state, allowing them to be learned and updated during the training process. This is a step towards a more trainable and flexible attention module, demonstrating how to make the mechanism learn from data.

* **Self attention Version 2 (nn.Linear)**
Building on the previous version, this implementation replaces the manually defined `nn.Parameter` tensors with `nn.Linear` layers. This is the more conventional and efficient way to implement the **Q**, **K**, and **V** projections in PyTorch. The `nn.Linear` layers automatically manage the weights and biases, streamlining the code and leveraging PyTorch's optimized linear algebra operations.

* **Casual attention**
Causal attention is a crucial variant used in language generation models like GPT. It introduces a mask to the attention mechanism that prevents a token from attending to future tokens in the sequence. This ensures that the model can only use past and present information to predict the next token, which is essential for tasks like text generation and autoregressive modeling. It also adds the concept of **dropouts** to prevent **overfitting**.

* **Multi head attention "wrapper"**
This is a wrapper class that encapsulates the multi-head attention logic. Instead of a single casual attention head, it runs several in parallel, each learning a different aspect of the relationships within the sequence. This "wrapper" organizes these separate heads and concatenates their outputs, allowing the model to capture richer, more diverse information about the input.

* **Multi head attention**
This is the full implementation of the multi-head attention mechanism. It combines the functionality of the single casual-attention head with the multi-head wrapper. By using multiple attention heads in parallel, this mechanism provides the model with the ability to focus on different parts of the input sequence simultaneously, significantly improving its ability to learn complex relationships and long-range dependencies.