#Data & Tokenizer
#kiểm tra độ dài văn bản:119003
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(text[:500])
print("Độ dài văn bản:", len(text))

# Tạo vocabulary:151 ký tự khác nhau 
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

print(chars)
print("Kích thước vocabulary:", vocab_size)
#Data Loader
#chuyển qua lại giữa số và ký tự 
print(stoi)
print(itos)
#chuyển qua lại giữa số và ký tự 
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("Xin chào"))
print(decode(encode("Xin chào")))

import torch
# biến ký tự thành số, biến danh sách txt thành Tensor :119003
data = torch.tensor(encode(text), dtype=torch.long)

print(data.shape)
print(data[:100])

#chia dữ liệu train/Validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print("Train:", train_data.shape)
print("Validation:", val_data.shape)

#tạo Batch
torch.manual_seed(1337)
#mỗi lần lấy 4 đoạn dữ liệu, mỗi đoạn có 8 token
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# xb.shape = torch.Size([4,8]]), y tương tự
xb, yb = get_batch('train')
#ví dụ: x: A B: y: B C, nhìn A đoán B, next-token prediction 
print('inputs:')
print(xb.shape)
print(xb)

print('targets:')
print(yb.shape)
print(yb)