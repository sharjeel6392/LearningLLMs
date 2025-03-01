import tiktoken
from importlib.metadata import version

print("tiktoken version: ", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."

ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(ids)

text = tokenizer.decode(ids)

print(text)

text = "Akwirw ier"
ids = tokenizer.encode(text)

for id in ids:
    print(tokenizer.decode([id]))

print(tokenizer.decode(ids))


''' 

BPE:
It builds it vocabulary by iteratively merging:
    frequent characters into subwords and
    frequent subwords into words.

For example:
    BPE starts with adding all single characters to its vocabulary ("a", "b", etc.)
    In the next stage, it merges character combinations that frequently occur together into subwords. For instance, "d" and "e" (define, depend, made, hidden, etc.)

The merges are determined by a "FREQUENCY CUTOFF"

'''