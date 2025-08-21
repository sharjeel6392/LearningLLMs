import torch
import torch.nn as nn

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False) -> None:
        """
            A construtor of SelfAttention class. It initializes the query, key and value tensors with nn.Linear method.
            
            Parameters
            -----------
            d_in :  int
                    The size of the input feature vector per token.
            d_out:  int
                    The size of the transformed feature vector after applying the projection

            Returns
            -------
                None
        """
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_keys = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_values = nn.Linear(d_in, d_out, bias = qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward phase calculation of W_query, W_keys, W_values

            Parameters
            ----------
            x : torch.Tensor, dtype = torch.float32, shape = (seq_len, d_in)
                Input Embeddings
            
            Returns
            -------
            context_vec: torch.Tensor
                Context vector which is the Attention output 
        """

        if x.dtype != torch.float32:
            raise TypeError(f"Expected x.dtype=torch.float32 but got {x.dtype}")

        keys = self.W_keys(x)
        queries = self.W_query(x)
        values = self.W_values(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim= -1)
        context_vec = attn_weights @ values

        return context_vec