import re


# read 'the-verdict.txt

with open("./Tokenizer/the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


# tokenize the words
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]


# convert tokens into token IDs
# adding special context tokens
# - <|unk|> token if the tokenizer encounters a word that is not part of the vocabulary
# - <|endoftext|> token to flag that the text sources that are concatenated for training are unrelated.
all_tokens = sorted(list(set(preprocessed))) # unique tokens
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_tokens)

vocab = {token:integer for integer, token in enumerate(all_tokens)}

# some other types of special context tokens include:
# |BOS| (beginning of sequence)
# |EOS| (end of sequence)
# |PAD| (padding)