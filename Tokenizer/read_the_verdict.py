import re


# read 'the-verdict.txt

with open("./Tokenization/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


# tokenize the words
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]


# convert tokens into token IDs
all_words = sorted(set(preprocessed)) # unique tokens
vocab_size = len(all_words)

vocab = {token:integer for integer, token in enumerate(all_words)}