import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        """
            A construtor of SelfAttention class. It initializes the query, key and value tensors with nn.Parameters method.
            
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
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_keys = nn.Parameter(torch.rand(d_in, d_out))
        self.W_values = nn.Parameter(torch.rand(d_in, d_out))

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

        keys = x @ self.W_keys
        queries = x @ self.W_query
        values = x @ self.W_values

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim= -1)
        context_vec = attn_weights @ values

        return context_vec