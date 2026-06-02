import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Clean Hyperparameters for TauCosNet ---
BATCH_SIZE = 128          # Stabilized batch size for efficient GPU memory alignment
BLOCK_SIZE = 256          # Full context window
MAX_ITERS = 50000         # Total training iterations
EVAL_INTERVAL = 200       
LEARNING_RATE = 5e-4      # Slightly higher starting rate since Cosine Attention is highly stable
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 20
EMBED_DIM = 384           
NUM_HEADS = 6             # 384 // 6 = 64 dimension per head
NUM_BLOCKS = 6            
DROPOUT = 0.2             

# --- Your Tau Theory Variable ---
TAU_TEMPERATUR = 0.1      # Lower = Laser focus (k-NN style), Higher = Global context floodlight
CHECKPOINT_DIR = "tau_checkpoints"
# --------------------------------------------

torch.manual_seed(1337)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "taucos_latest.pth")


def save_checkpoint(model, optimizer, iter_step, path):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'iter': iter_step,
        'tau_temperatur': TAU_TEMPERATUR,
    }
    torch.save(checkpoint, path)
    print(f"[Checkpoint Saved] iteration {iter_step} -> {path}")


def load_checkpoint(model, optimizer, path):
    if not os.path.exists(path):
        return 0
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_iter = checkpoint.get('iter', 0)
    print(f"[Checkpoint Loaded] Resuming from iteration {start_iter} at '{path}'")
    return start_iter

# --- 1. Data Processing ---
file_path = '/home/kuba/workspace/Transformer/tau/input.txt'
if not os.path.exists(file_path):
    print(f"Error: '{file_path}' not found. Please verify the directory.")
    exit()

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    split_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(split_data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([split_data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([split_data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- 2. Tau-Cosine Attention Engine ---
class TauCosineHead(nn.Module):
    """ One Head of Cosine Attention governed by Temperature Tau """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.query = nn.Linear(EMBED_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_size, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        
        # Project tokens to raw matrices
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # CRITICAL CORE STEP: L2-Normalization across the channel dimension
        # This forces all vectors onto a unit hypersphere where length is strictly 1.0
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)

        # Compute raw Cosine Similarities via matrix multiplication (Values are locked between -1 and 1)
        cosine_sim = q_norm @ k_norm.transpose(-2, -1) # (B, T, T)

        # Apply your Tau Temperature valve to scale the angular boundaries
        scaled_sim = cosine_sim / TAU_TEMPERATUR

        # Causal masking (no looking into the future)
        wei = scaled_sim.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Softmax turns the scaled cosines into a clean probability map
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Extract features from the Value matrix
        out = wei @ v # (B, T, head_size)
        return out

class MultiHeadTauCosineAttention(nn.Module):
    """ Combines multiple Tau-Cosine attention heads in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([TauCosineHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# --- Standard Transformer Subcomponents ---
class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(), # Upgraded to GELU for smoother gradient landscapes
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(DROPOUT),
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_size = embed_dim // num_heads
        self.sa = MultiHeadTauCosineAttention(num_heads, head_size)
        self.ffwd = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --- 3. The Core Network ---
class TauCosNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.blocks = nn.Sequential(*[TransformerBlock(EMBED_DIM, num_heads=NUM_HEADS) for _ in range(NUM_BLOCKS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- 4. Initialization & Setup ---
model = TauCosNet().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
start_iter = load_checkpoint(model, optimizer, CHECKPOINT_PATH)

print(f"TauCosNet initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M Parameters running on {DEVICE.upper()}.")

train_losses, val_losses, steps = [], [], []

# --- 5. Clean Training Loop ---
if start_iter < MAX_ITERS:
    print(f"\n--- Training Loop Active [Tau = {TAU_TEMPERATUR}] ---")
    if start_iter > 0:
        print(f"Resuming from checkpoint at iteration {start_iter}.")
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    model.train()
    current_iter = start_iter

    try:
        for iter in tqdm(range(start_iter, MAX_ITERS), desc="Processing Batches"):
            current_iter = iter
            
            if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
                losses = estimate_loss(model)
                steps.append(iter)
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])
                print(f"\nIteration {iter}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")
                
                ax.clear()
                ax.plot(steps, train_losses, label='Train Loss')
                ax.plot(steps, val_losses, label='Val Loss')
                ax.set_title(f"TauCosNet Learning Graph (Tau: {TAU_TEMPERATUR})")
                ax.legend(); ax.grid(True)
                plt.pause(0.1); plt.show()

            xb, yb = get_batch('train')
            _, loss = model(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (iter + 1) % 5000 == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f'taucos_model_{iter+1}.pth')
                save_checkpoint(model, optimizer, iter + 1, ckpt_path)
                save_checkpoint(model, optimizer, iter + 1, CHECKPOINT_PATH)

            if (iter + 1) % 500 == 0:
                model.eval()
                with torch.no_grad():
                    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
                    snippet = decode(model.generate(context, max_new_tokens=50)[0].tolist())
                    print(f"\n--- Snippet bei Schritt {iter+1} ---")
                    print(snippet.replace('\n', ' ')[:100] + "...")
                model.train()

    except KeyboardInterrupt:
        print("\nProcess manually halted by user.")
    finally:
        save_checkpoint(model, optimizer, current_iter + 1 if current_iter < MAX_ITERS else MAX_ITERS, CHECKPOINT_PATH)
        plt.ioff()
else:
    print(f"\nCheckpoint already reached MAX_ITERS ({MAX_ITERS}). No training required.")

# --- 6. Final Model Execution ---
print("\n--- Output Generation via TauCosNet ---")
model.eval()
with torch.no_grad():
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
plt.show()
