import torch
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89], # Your      (x^1)
        [0.55, 0.87, 0.66], # journey   (x^2)
        [0.57, 0.85, 0.64], # starts    (x^3)
        [0.22, 0.58, 0.33], # with      (x^4)
        [0.77, 0.25, 0.10], # one       (x^5)
        [0.05, 0.80, 0.55]  #step       (x^6)
    ]
)

# lets say we want to compute z(2), that is, inputs[1] is the query input
# query = inputs[1]
# attn_scores_2 = torch.empty(inputs.shape[0])
# for i, x_i in enumerate(inputs):
#     attn_scores_2[i] = torch.dot(x_i, query)

# print(f'Attention scores w2i =  {attn_scores_2}')

# attn_weights_2 = torch.softmax(attn_scores_2, dim = 0)
# print(f'Attention weights, a2i = {attn_weights_2}')
# print(f'Attention weights sun = {attn_weights_2.sum()}')

# context_vec_2 = torch.zeros(query.shape)
# for i, x_i in enumerate(inputs):
#     context_vec_2 += attn_weights_2[i] * x_i

# print(f'Context vector, z(2) = {context_vec_2}')


attn_scores = inputs @ inputs.T
print(f'Attention scores: \n{attn_scores}')

attn_weights = torch.softmax(attn_scores, dim = -1)
print(f'Attention weights: \n{attn_weights}')

all_context_vecs = attn_weights @ inputs
print(f'All context vectors: \n{all_context_vecs}')