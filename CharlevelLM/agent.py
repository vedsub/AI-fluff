import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# For reproducibility
gen = torch.Generator().manual_seed(2147483647)

# 1. LOAD DATA
_dir = os.path.dirname(os.path.abspath(__file__))
# Check if data.txt exists, otherwise create dummy data for testing
data_path = os.path.join(_dir, 'data.txt')
if not os.path.exists(data_path):
    print("data.txt not found. Using dummy data.")
    words = ["hello", "world", "machine", "learning", "neural", "network"]
else:
    words = open(data_path, 'r').read().splitlines()

words = [w for w in words if w]
text = ''.join(words)

# 2. BUILD VOCABULARY
chars = sorted(list(set(text)))
chars = ['.'] + chars # Ensure '.' is at index 0
vocab_size = len(chars)
ctoi = {ch: i for i, ch in enumerate(chars)}
itoc = {i: ch for ch, i in ctoi.items()}

print(f"Vocab size: {vocab_size}")

# 3. DEFINE THE NEW MODEL (PART 2 VERSION)
class MLP_Part2(nn.Module):
    def __init__(self, vocab_size, block_size=3, embed_dim=10, hidden_dim=200):
        super().__init__()
        self.block_size = block_size
        self.embed_dim = embed_dim
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # Layers: Linear -> BatchNorm -> Tanh
        self.linear1 = nn.Linear(embed_dim * block_size, hidden_dim, bias=False) # Bias not needed with BatchNorm
        self.bn1 = nn.BatchNorm1d(hidden_dim) 
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

        # --- PART 2: SMART INITIALIZATION ---
        with torch.no_grad():
            # 1. Kaiming Init
            self.linear1.weight *= (5/3) / ((embed_dim * block_size)**0.5)
            
            # 2. Fix "Hockey Stick" Loss (make initial weights tiny)
            self.linear2.weight *= 0.01 
            self.linear2.bias *= 0 

    def forward(self, x, targets=None):
        emb = self.embed(x) # (B, block_size, embed_dim)
        x = emb.view(-1, self.embed_dim * self.block_size)
        
        x = self.linear1(x)
        x = self.bn1(x) 
        x = self.tanh(x)
        
        logits = self.linear2(x)
        
        if targets is None:
            return logits, None
        
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, itoc, ctoi, max_new_tokens=200):
        self.eval() # IMPORTANT: Switch to eval mode for BatchNorm
        context = [ctoi['.']] * self.block_size
        out = []
        for _ in range(max_new_tokens):
            x = torch.tensor([context])
            logits, _ = self(x)
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1).item()
            
            context = context[1:] + [ix]
            out.append(itoc[ix])
            if itoc[ix] == '.': break
            
        self.train() # Switch back to train mode
        return ''.join(out)

# 4. PREPARE DATASET
# 4. PREPARE DATASET (SPLIT INTO TRAIN / DEV / TEST)
block_size = 3

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [ctoi['.']] * block_size
        for ch in w + '.':
            X.append(context)
            Y.append(ctoi[ch])
            context = context[1:] + [ctoi[ch]]
    return torch.tensor(X), torch.tensor(Y)

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])     # 80% Training
Xdev, Ydev = build_dataset(words[n1:n2]) # 10% Validation (Dev)
Xte, Yte = build_dataset(words[n2:])     # 10% Test

print(f"Train size: {len(Xtr)} | Val size: {len(Xdev)} | Test size: {len(Xte)}")

# Training Loop
# 5. INSTANTIATE AND TRAIN
model = MLP_Part2(vocab_size, block_size=block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

print("Starting training...")
for step in range(10000): # Increased steps for better convergence
    # CRITICAL: Sample only from Xtr (Training Set)
    idxs = torch.randint(0, len(Xtr), (32,))
    xb, yb = Xtr[idxs], Ytr[idxs]
    
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Decay learning rate at step 5000
    if step == 5000:
        for g in optimizer.param_groups:
            g['lr'] = 0.01
            
    if step % 1000 == 0:
        print(f"step {step:5d} | train loss {loss.item():.4f}")

# 6. EVALUATION
@torch.no_grad()
def split_loss(split):
    model.eval() # Switch to eval mode
    x, y = {
        'train': (Xtr, Ytr),
        'val':   (Xdev, Ydev),
        'test':  (Xte, Yte),
    }[split]
    logits, loss = model(x, y)
    print(f"{split} loss: {loss.item():.4f}")
    model.train() # Switch back to train mode

print("\n--- Evaluation ---")
split_loss('train')
split_loss('val') 
# If 'val' loss is much higher than 'train' loss, you are overfitting.

# 6. GENERATE
print("\nGenerated Output:")
# Note: We now pass itoc and ctoi as required by the new class method
print(model.generate(itoc, ctoi, max_new_tokens=200))