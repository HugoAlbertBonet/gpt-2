import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

class tokenizer:
    def __init__(self, chars):
        self.stoi = { ch:i for i, ch in enumerate(chars) }
        self.itos = { i:ch for i, ch in enumerate(chars) }

    def encode(self, s: str) -> list:
        return [self.stoi[c] for c in s]
    
    def decode(self, l: list) -> str:
        return "".join([self.itos[i] for i in l])
    

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #lookup table for the logits of the next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        #idx and targets are (Batch B, Block T) tensors of integers
        logits = self.token_embedding_table(idx) # (Batch B, Block T, Vocab_size C), each index will extract the row corresponding to it

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim = -1) #(B, C)
            idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim= 1) # (B, T+1)

        return idx


def create_vocabulary(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    return chars, vocab_size

def get_batch(split, block_size, batch_size, seed = 42):
    torch.manual_seed(seed)
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

if __name__ == "__main__":

    # TOKENIZE THE TEXT
    with open("input.txt", "r") as f:
        text = f.read()
    chars, vocab_size = create_vocabulary(text)

    encoder = tiktoken.get_encoding("gpt2")
    
    tok = tokenizer(chars)

    data = torch.tensor(tok.encode(text), dtype = torch.long)

    # TRAIN/TEST SPLIT
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]


    
    batch_size = 4 
    block_size = 8 

    xb, yb = get_batch("train", block_size, batch_size, seed = 1337)
    print(xb)

    m = BigramLanguageModel(vocab_size)
    logits, loss = m(xb, yb)

    idx = torch.zeros((1,1), dtype = torch.long) # 1 batch of 1 character with idx 0 (new line)
    print(tok.decode(m.generate(idx, max_new_tokens= 100)[0].tolist()))


    #TRAIN THE BIGRAM MODEL
    optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

    batch_size = 32

    for steps in range(10000):
        xb, yb = get_batch("train", block_size, batch_size)

        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()

    print(loss.item())

    print(tok.decode(m.generate(idx, max_new_tokens= 300)[0].tolist()))



    
