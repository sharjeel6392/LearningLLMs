import data_loader

with open("./Tokenizer/the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataLoader = data_loader.create_dataloader_v1( raw_text, batch_size = 8, max_length = 4, stride  =4, shuffle = False)
data_iter = iter(dataLoader)
inputs, targets = next(data_iter)

print("Inputs: \n", inputs)
print("TargetsL \n", targets)