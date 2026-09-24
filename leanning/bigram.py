# Bigram Language Model

import torch
import torch.nn as nn
from torch.nn import functional as F

from data import *


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
# tức là mỗi token được biến thành một vector có 151 chiều.
m = BigramLanguageModel(vocab_size)

logits, loss = m(xb, yb)

#generate
def generate(idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, loss = m(idx)

        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


#thêm optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
#vòng lặp trainning. Hiểu đơn giản là: lấy dữ liệu, model dự đoán, tính loss, xóa gradient cũ, backward() tính gradient, optinizer.step() cập nhật model,  lặp lại 
for steps in range(1000):
    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if steps % 100 == 0:
        print(steps, loss.item())
#kiểm tra sau khi model học 
context = torch.zeros((1, 1), dtype=torch.long)

generated = generate(context, max_new_tokens=100)
