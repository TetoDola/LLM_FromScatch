import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import os

# Config
num_epochs = 3
batch_size = 64
block_size = 256
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_batches = 50
eval_interval = 500  # how many steps between validations
save_interval = 500  # how many steps between checkpoints
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
seed = 1337
best_val_loss = float('inf')

torch.manual_seed(seed)

# Data Loading
with open('all_epubs.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Character set / vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(ix_list):
    return ''.join(itos[i] for i in ix_list)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # 90% train, 10% val
train_data = data[:n]
val_data = data[n:]


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        super().__init__()
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return x, y


train_dataset = CharDataset(train_data, block_size)
val_dataset = CharDataset(val_data, block_size)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=True)

# Model Definition
class Head(nn.Module):
    # A single attention head.
    def __init__(self, head_size, block_size, n_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Mask to ensure we only attend to the left in the sequence
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.shape[-1]))  # (B, T, T)
        mask = self.tril[:T, :T] == 0
        wei = wei.masked_fill(mask, float('-inf'))

        # softmax
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, block_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, block_size, n_embd, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, block_size, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()

        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Embed tokens and positions
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb.unsqueeze(0)  # (B, T, n_embd)

        # Transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)

        # Final language modeling head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten the logits and targets to compute cross entropy
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(
            self,
            idx,
            max_new_tokens,
            temperature=1.0,
            top_k=None,
            top_p=None
    ):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            if top_k is not None:
                v, ix = torch.topk(logits, top_k)
                out = torch.full_like(logits, float('-inf'))
                out.scatter_(1, ix, v)
                logits = out
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                for batch_i in range(logits.size(0)):
                    indices_to_remove = sorted_indices[batch_i, sorted_indices_to_remove[batch_i]]
                    logits[batch_i, indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

model = GPTLanguageModel(
    vocab_size, n_embd, n_head, n_layer, block_size, dropout
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")


# Evaluation Helpers
@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (xb, yb) in enumerate(data_loader):
        if i >= eval_batches:  # limit number of validation batches
            break
        xb = xb.to(device)
        yb = yb.to(device)
        _, loss = model(xb, yb)
        batch_size, block_sz = xb.shape
        total_loss += loss.item() * batch_size * block_sz
        total_tokens += batch_size * block_sz
    avg_loss = total_loss / (total_tokens if total_tokens > 0 else 1)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

# Training Loop
step = 0
for epoch in range(num_epochs):
    print(f"--- Epoch {epoch + 1}/{num_epochs} ---")
    model.train()

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Optional gradient clipping
        # nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        # scheduler.step()  # if using a scheduler

        step += 1
        if step % 100 == 0:
            print(f"Step {step}, loss: {loss.item():.4f}")

        # Periodic evaluation
        if step % eval_interval == 0:
            val_loss, val_ppl = evaluate(model, val_loader)
            print(f"Validation Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")

            # Save "best" checkpoint
            global best_val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("New best model found! Saving checkpoint...")
                torch.save(model.state_dict(), "best_model.pth")

        # Periodic saving
        if step % save_interval == 0:
            ckpt_path = f"checkpoint_step_{step}.pth"
            print(f"Saving checkpoint to {ckpt_path}...")
            torch.save(model.state_dict(), ckpt_path)
# Final Evaluation
val_loss, val_ppl = evaluate(model, val_loader)
print(f"Final Validation Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")

model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # single token '0'
generated_idx = model.generate(
    context,
    max_new_tokens=200,
    temperature=1.0,
    top_k=50,
    top_p=None
)
generated_str = decode(generated_idx[0].tolist())
print("Generated text:")
print(generated_str)
