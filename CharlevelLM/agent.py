import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# For reproducibility (same seed as article)
gen = torch.Generator().manual_seed(2147483647)

# Load data (article uses Michael Jackson lyrics)
words = open('data.txt', 'r').read().splitlines()   # better than .split() for lines
text = ''.join(words)   # or keep as list of words — both work

# Vocabulary (same as article)
chars = sorted(list(set(text)))
vocab_size = len(chars)
ctoi = {ch: i for i, ch in enumerate(chars)}
itoc = {i: ch for ch, i in ctoi.items()}

print(f"Vocab size: {vocab_size} chars → {''.join(chars)}")
# Build count matrix (exactly like article)
counts = torch.zeros((vocab_size, vocab_size), dtype=torch.float32)
for word in words:
    chs = ['.'] + list(word) + ['.']   # article sometimes uses start/end (optional)
    for ch1, ch2 in zip(chs, chs[1:]):
        counts[ctoi[ch1], ctoi[ch2]] += 1

# Smoothing + normalization (article uses +1)
counts += 1
probs = counts / counts.sum(dim=1, keepdim=True)   # shape (vocab_size, vocab_size)

# Generate (same logic as article)
def generate_count_based(model_probs, max_len=50, seed=2147483647):
    torch.manual_seed(seed)
    idx = torch.tensor([ctoi['.']], dtype=torch.long)   # start token
    result = []
    
    for _ in range(max_len):
        p = model_probs[idx[-1]]
        next_idx = torch.multinomial(p, num_samples=1, generator=gen).item()
        result.append(itoc[next_idx])
        idx = torch.tensor([next_idx])
        if itoc[next_idx] == '.': break   # optional stop
    
    return ''.join(result)

print("Sample:", generate_count_based(probs, max_len=200))

log_lik = 0.0
n = 0
for word in words[:100]:   # or all words
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        p = probs[ctoi[ch1], ctoi[ch2]]
        log_lik += torch.log(p).item()
        n += 1

print(f"Avg NLL (count-based): {-log_lik / n:.4f}")
class NeuralBigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)   # logits table

    def forward(self, idx, targets=None):
        logits = self.embedding(idx)           # (B,) or (B,T) → (B, vocab_size) or (B,T,vocab_size)
        if targets is None:
            return logits, None
        
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, max_new_tokens=200, temperature=1.0):
        idx = torch.zeros((1, 1), dtype=torch.long)   # start token
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1, generator=gen)
            idx = torch.cat((idx, next_idx), dim=1)
        return ''.join(itoc[i.item()] for i in idx[0])

# Data (context = previous char)
xs, ys = [], []
for word in words:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(ctoi[ch1])
        ys.append(ctoi[ch2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)

model = NeuralBigram(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)   # article often uses higher LR here

for step in range(10000):
    logits, loss = model(xs, ys)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 1000 == 0:
        print(f"step {step:5d} | loss {loss.item():.4f}")

print("Generated:", model.generate(max_new_tokens=300, temperature=0.9))

class TrigramMLP(nn.Module):
    def __init__(self, vocab_size, block_size=3, embed_dim=32, hidden_dim=200):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * block_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x, targets=None):
        # x: (B, block_size)
        emb = self.embed(x)                     # (B, block_size, embed_dim)
        x = emb.view(-1, embed_dim * self.block_size)  # flatten
        logits = self.net(x)                    # (B, vocab_size)
        
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, max_new_tokens=200, temperature=1.0):
        context = torch.zeros((1, self.block_size), dtype=torch.long)
        out = []
        for _ in range(max_new_tokens):
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1, generator=gen)
            context = torch.cat((context[:, 1:], idx_next), dim=1)
            out.append(itoc[idx_next.item()])
        return ''.join(out)

# Build dataset (context of block_size previous chars)
block_size = 3
X, Y = [], []
for word in words:
    context = [ctoi['.']] * block_size
    for ch in word + '.':
        X.append(context)
        Y.append(ctoi[ch])
        context = context[1:] + [ctoi[ch]]

X = torch.tensor(X)
Y = torch.tensor(Y)

model = TrigramMLP(vocab_size, block_size=block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Mini-batch training
for step in range(30000):
    idxs = torch.randint(0, len(X), (256,))
    xb, yb = X[idxs], Y[idxs]
    
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 2000 == 0:
        print(f"step {step:5d} | loss {loss.item():.4f}")

print("Generated:", model.generate(max_new_tokens=400, temperature=0.85))