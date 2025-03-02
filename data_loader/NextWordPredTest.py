import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

with open("./Tokenizer/the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

encoded_text = tokenizer.encode(raw_text)

#print(len(encoded_text))

encoded_sample = encoded_text[50:]

'''
    To create the input-target pairs for the "next word prediction" task, we define two variables:
        x: input tokens
        y: targets, which are inputs, shifted by 1
'''

# context_size determines how many tokens are included in the input
context_size = 4

x = encoded_sample[ : context_size]
y= encoded_sample[1 : context_size + 1]

print(f"x:   {x}")
print(f"y:   {y}")

for i in range(1, context_size + 1):
    context = encoded_sample[:i]
    desired = encoded_sample[i]
    #print(context, " ----> ", desired)
    print(tokenizer.decode(context), " ----> ", tokenizer.decode([desired]))
