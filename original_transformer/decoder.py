import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

#HYPERPARAMETERS
batch_size = 32
block_size = 512
max_iters = 4000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 384 # every head is 64 dim
dropout = 0.2
n_head = 6
n_layer = 6
torch.manual_seed(1337)


class tokenizer:
    def __init__(self, chars):
        self.stoi = { ch:i for i, ch in enumerate(chars) }
        self.itos = { i:ch for i, ch in enumerate(chars) }

    def encode(self, s: str) -> list:
        return [self.stoi[c] for c in s]
    
    def decode(self, l: list) -> str:
        return "".join([self.itos[i] for i in l])
    

class Head(nn.Module):
    """ One head of self-attention"""
    """ For cross-attention the query and value apply to the input and the key to the output """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))) #because it is not a parameter

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x): 
        B, T, C = x.shape
        key = self.key(x)
        query = self.query(x)

        wei = query @ key.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out
    
class CrossAttentionHead(nn.Module):
    """ One head of cross-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))) #because it is not a parameter

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, x): 
        B, T, C = x.shape
        key = self.key(x)
        query = self.query(input)

        wei = query @ key.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out
    

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out
    

class FeedForward(nn.Module):
    """ a simple linear liyer followed by a non-linearity"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4* n_embed),
            nn.ReLU(),
            nn.Linear(4* n_embed, n_embed), #return to residual pathway
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication (attention) followed by computation"""

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #per-token normalization
        x = x + self.ffwd(self.ln2(x)) #residual connections
        return x


class DecoderOnly(nn.Module):
    """ Decoder-only transformer with character tokenization,
    for encoder-decoder a cross-attention layer should be added"""
    def __init__(self):
        super().__init__()
        #lookup table for the logits of the next token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape

        #idx and targets are (Batch B, Block T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # (Batch B, Block T, n_embed C), each index will extract the row corresponding to it
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T, C)
        x = tok_emb + pos_emb #(B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) #(Batch, Block, Vocab_size)
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
            idx_cond = idx[:, -block_size:] #crop to not exceed the block size
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim = -1) #(B, C)
            idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim= 1) # (B, T+1)

        return idx
    
class Encoder(nn.Module):
    """ Decoder-only transformer with character tokenization,
    for encoder-decoder a cross-attention layer should be added"""
    def __init__(self):
        super().__init__()
        #lookup table for the logits of the next token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.block = Block(n_embed, n_head=1)

    def forward(self, idx, targets = None):
        B, T = idx.shape

        #idx and targets are (Batch B, Block T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # (Batch B, Block T, n_embed C), each index will extract the row corresponding to it
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T, C)
        x = tok_emb + pos_emb #(B,T,C)
        x = self.block(x)
        return x


def create_vocabulary(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    return chars, vocab_size

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() #sets to evaluation phase, with our model it does nothing
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() #sets to training phase, with our model it does nothing
    return out

if __name__ == "__main__":
    print("Running on", device)

    with open("aitana.txt", "r", encoding = "utf-8") as f:
        text = f.read()

    chars, vocab_size = create_vocabulary(text)

    tok = tokenizer(chars)

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    #MODEL CREATION
    model = DecoderOnly()
    m = model.to(device)
    print("The model has", sum(p.numel() for p in m.parameters())/1e6, "M parameters")
    #TRAIN THE BIGRAM MODEL
    optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train_loss {losses['train']:4f}, val_loss {losses['val']:4f}")
        xb, yb = get_batch("train")

        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()

    print(loss.item())

    context = torch.zeros((1,1), dtype = torch.long, device = device) # 1 batch of 1 character with idx 0 (new line)
    print(tok.decode(m.generate(context, max_new_tokens= 3000)[0].tolist()))
    torch.save(m, "aitana.pt")




    
