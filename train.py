import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 64
context_length = 256
max_iters = 1
learning_rate = 3e-4
eval_interval = 500
eval_iters = 200
n_embed = 384
n_layers = 6
n_heads = 6
dropout = 0.2
max_new_tokens = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(32)

# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open("data/input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)

idx_to_char = {idx: char for idx, char in enumerate(chars) }
char_to_idx = {char: idx for idx, char in enumerate(chars) }

encode = lambda x: [char_to_idx[i] for i in x]
decode = lambda x: ''.join([idx_to_char[i] for i in x])

# print((encode(text[:1000])))


data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

x = train_data[: context_length]
y = train_data[1:context_length + 1]

for i in range(context_length):
    context = x[:i+1]
    target = y[i]
    print(f'for context ', context, '- target ',target)


def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - context_length, size = (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in idx])
    y = torch.stack([data[i+1: i+context_length + 1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for j in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[j] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

xb, yb = get_batch('train')

print('inputs:')
print(xb.shape)
# print(xb)
print('targets:')
print(yb.shape)
# print(yb)

# print('----')

# for b in range(batch_size):
#     for i in range(context_length):
#         context = xb[b, :i + 1]
#         target = yb[b, i]
#         print(f'When context = {context}, then target = {target}')

class Head(nn.Module):
    """Single head of """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        key = self.key(x) # (B,T,C); C/4 because we are keeping head size same as n_embed (for now)
        query = self.query(x) # (B,T,C)
        value = self.value(x) # (B,T,C)

        weights = query @ key.transpose(-2,-1) * C**-0.5 # # (B,T,C) * # (B,C,T) = (B,T,T)
        weights = weights.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        weights = F.softmax(weights, dim = -1) # (B,T,T)
        weights = self.dropout(weights)
        out = weights @ value # (W, T, C) 
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.projection(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, 4 * n_embed), 
                                 nn.ReLU(),
                                 nn.Linear(4 * n_embed, n_embed),
                                 nn.Dropout(dropout))
        
    def forward(self, x):
        out = self.net(x)
        return out

class Block(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.head_size = n_embed//n_heads
        self.self_attention = MultiHeadAttention(self.n_heads, self.head_size)
        self.ffwd = FeedForward(self.n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BiGramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding_table = nn.Embedding(context_length, n_embed)
        self.attention_ffwd_blocks = nn.Sequential(*[Block(n_embed, n_heads=n_heads) for _ in range(n_layers)]) 
        self.layernorm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        

    def forward(self, idx, target = None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx) # (B, T, n_embed);;   x: (B, T)
        pos_emb = self.pos_embedding_table(torch.arange(T, device = device)) # (T, n_embed)
        x = token_emb + pos_emb # (B, T, n_embed)
        x = self.attention_ffwd_blocks(x)
        x = self.layernorm(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if target is None:
            return logits, None

        else: # Else block for generation ie target isn't available
            B, T, C = logits.shape # original shape; C is the embedding size
            logits = logits.view(B*T, C) # Reshaping because torch's cross entropy loss function expects tensors in a certain format; 
            target = target.view(B * T) # reshaping due to the reason above
            loss = F.cross_entropy(logits, target)
            return logits, loss

    def generate(self, x, max_new_tokens = 100):
        # x is of shape (B, T) array of indices in the current context
        for i in range(max_new_tokens):
            x_cond = x[..., -context_length:] # Because if the input is more than context length, the attention won't work
            logits, loss = self(x_cond  )
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim = 1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            x = torch.cat((x, idx_next), dim = 1) # (B, T+1) as we concat B, T with B, 1 on axis = 1

        return x

model = BiGramModel()
model = model.to(device)

logits, loss = model(xb, yb)

print(logits.shape)
print(loss)

# x_ = torch.zeros(size = (1,1), dtype=torch.long) # Creating a zero tensor as initial token
# print(decode(model.generate(x_, max_new_tokens=100)[0].tolist())) # generating the text

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for i in range(max_iters):
    
    if i % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {i}: train loss: {losses['train']:.4f}, validation loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss.item())

PATH = "model.pt"
torch.save(model.state_dict(), PATH)