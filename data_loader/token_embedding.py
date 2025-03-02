import torch
from data_loader import create_dataloader_v1

with open("./Tokenizer/the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

vocab_size = 50257 # the vocab size of BPE Tokenizer
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size = 8, max_length = max_length, stride = max_length, shuffle = False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# token embeddings

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

context_length = max_length

# positional Encoding

pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

print(pos_embeddings.shape)

# input embeddings = token embeddings + positional embeddings

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)