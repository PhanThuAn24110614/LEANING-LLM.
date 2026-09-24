import torch
import torch.nn as nn
from torch.nn import functional as F

from data import *
#Self-Attention sẽ cho phép một token nhìn lại các token trước nó.
xb, yb = get_batch('train')
#khai báo t riêng, thêm sau
B, T = xb.shape

# chạy ra kết quả [4,8,32] 4 đoạn trong batch, mỗi đoạn 8 token, mỗi token biển diwwnx bằng vecto 32 chiều
n_embd = 32
#thêm vào sau
head_size = 16
token_embedding_table = nn.Embedding(vocab_size, n_embd)

tok_emb = token_embedding_table(xb)
#Thiết lập một Cơ chế Tự chú ý đơn lẻ (Single Self-Attention Head)
#Single Head thành class
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float('-inf')
        )

        wei = F.softmax(wei, dim=-1)

        out = wei @ v

        return out

#Nâng cấp lên Cơ chế Chú ý đa đầu (Multi-Head Attention)
#không hiểu cái gì cảaaaaaa
num_heads = 4
head_size = 8
#
heads = nn.ModuleList(
    [Head(head_size) for _ in range(num_heads)]
)

print(heads)

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
    num_heads = 4
head_size = 8

mha = MultiHeadAttention(num_heads, head_size)

out = mha(tok_emb)

print("Multi-Head output shape:", out.shape)

#Xây dựng Khối mạng nơ-ron chuyển tiếp (Feed-Forward Network)
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)

ffwd = FeedForward(n_embd)

out = ffwd(tok_emb)

print("Feed-Forward output shape:", out.shape)

#Transformer Block.
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)

        return x

#test
block = Block(n_embd=32, n_head=4)

out = block(tok_emb)

print("Transformer Block output shape:", out.shape)

#Lắp ghép toàn bộ mô hình (Assembling the Model)
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4)
        )

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)

        pos_emb = self.position_embedding_table(
            torch.arange(T)
        )

        x = tok_emb + pos_emb
        x = self.blocks(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

# test
m = GPTLanguageModel()

xb, yb = get_batch('train')

logits, loss = m(xb, yb)

print("GPT logits shape:", logits.shape)
print("Initial loss:", loss.item())

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
#training loop
for steps in range(1000):
    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if steps % 100 == 0:
        print(steps, loss.item())

#generate
def generate(idx, max_new_tokens):
    for _ in range(max_new_tokens):

        idx_cond = idx[:, -block_size:]

        logits, loss = m(idx_cond)

        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx

context = torch.zeros((1, 1), dtype=torch.long)

generated = generate(context, max_new_tokens=200)

print(decode(generated[0].tolist()))
