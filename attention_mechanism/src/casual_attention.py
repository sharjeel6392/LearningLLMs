import torch
import torch.nn as nn

class CasualAttention(nn.Module):
    """
    A casual self-attention module.

    This module applies a causal mask to the attention scores to ensure that each token 
    can only attend to previous tokens in the sequence. It's a fundamental
    building block in models like the GPT series.
    """
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, qkv_bias: bool = False):
        """
        Initializes the CasualAttention module.

        Parameters
        ----------
        d_in :          int
                        The dimension of the input tokens.
        d_out:          int
                        The dimension of the output context vectors.
        context_length: int
                        The maximum length of the input sequence.
        dropout:        The dropout rate to apply to the attention weights.
        qkv_bias:       bool, optional
                        If True, a bias term is added to the 
                        query, key, and value linear layers. 
                        Defaults to False.
        """
        super().__init__()
        self.dout = d_out
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1
            )
        )
    def forward(self, x: torch.Tensor):
        """
        Performs the forward pass of the attention mechanism.

        Parameters
        ----------

            x:      torch.Tensor
                    The input tensor of shape (batch_size, num_tokens, d_in).

        Returns
        -------
            Context_vector: torch.Tensor
                            The output context vector of shape (batch_size, num_tokens, d_out).
        """
        b, num_tokens, d_in = x.shape
        key = self.W_key(x)
        query = self.W_query(x)
        value = self.W_value(x)

        attn_scores = query @ key.transpose(1,2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / key.shape[-1] ** 0.5, dim = 1)
        attn_weights = self.dropout(attn_weights)

        context_vector = attn_weights @ value
        return context_vector
    

def main():

    """
    Runs an example of the CasualAttention class with parameterized inputs.

    This function sets up a sample input tensor, initializes the attention module,
    and demonstrates a forward pass. It's a convenient way to test the class.

    """

    x = torch.tensor(
        [
            [0.43, 0.15, 0.89], # Your     (x^1)
            [0.55, 0.87, 0.66], # journey  (x^2)
            [0.57, 0.85, 0.64], # starts   (x^3)
            [0.22, 0.58, 0.33], # with     (x^4)
            [0.77, 0.25, 0.10], # one      (x^5)
            [0.05, 0.80, 0.55] # step      (x^6)
        ]
    )
    d_in, d_out = x.shape[1], 2
    batch = torch.stack((x, x), dim = 0)
    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CasualAttention(d_in, d_out, context_length=context_length, dropout=0.0)
    context_vecs = ca(batch)

if __name__ == '__main__':
    main()