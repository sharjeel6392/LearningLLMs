import tiktoken
from importlib.metadata import version

print("tiktoken version: ", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."

ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(ids)

text = tokenizer.decode(ids)

print(text)

smallerText = "someunknownPlace"
testText = "some unknown Place"
print(tokenizer.encode(smallerText))
print(tokenizer.encode(testText))