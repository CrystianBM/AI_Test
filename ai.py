import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 20000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

# stoi = String to Int ///// itos = Int to String
# for each character in chars
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#encodes text and organize inside a torch.tensor
data = torch.tensor(encode(text), dtype=torch.long)

#splitting up data into train sets and validation sets
#90% will be train and the rest for validation
n_data = int(0.9*len(data))
train_data = data[:n_data]
val_data = data[n_data:]

#context and target
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):

        #idx and Targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            #unfold logits into B,T,C
            B, T, C = logits.shape
            #fold logits again turning it from tri-dimensional(B,T,C) to bi-dimensional(BT, C)
            logits = logits.view(B*T, C)
            #same to targets but needs to turn from bi-dimensional(B,T) to one-dimensional(BT)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) #cross_entropy needs the data in B,C,T

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx = (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #get predictions
            logits, loss = self(idx)
            #focus only on the last line step
            logits = logits[:, -1, :] #becomes (B,C)
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) #(B,C)
            #sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            #append sampled index to the runnning sequence
            idx = torch.cat((idx, idx_next), dim=1) #(B,T+1)
        return idx

xb, yb = get_batch('train')

model = BigramLanguageModel(vocab_size)
n_lang = model.to(device)
logits, loss = n_lang(xb, yb)

optimizer = torch.optim.AdamW(n_lang.parameters(), lr=learning_rate)

for iter in range(max_iters):

    #every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")
    #sample a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = n_lang(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(n_lang.generate(idx = context, max_new_tokens=300)[0].tolist()))