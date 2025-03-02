import dataLoader_class
import tiktoken

def create_dataloader_v1(txt, batch_size = 4, max_length = 256, stride = 1, shuffle = True, drop_last = True, num_workers = 0):
    
    # initializes the tokenizer
    
    tokenizer = tiktoken.get_encoding("gpt2")

    # creates the dataset

    dataset = dataLoader_class.GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = dataLoader_class.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )

    return dataloader