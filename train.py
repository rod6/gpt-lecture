import torch
from gpt import GPTLanguageModel, device, block_size, batch_size, text, encode, decode

max_iters = 200
eval_interval = 100
learning_rate = 1e-3
eval_iters = 100


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# 获取一个batch的数据，其中split是用于区分训练数据/评估数据
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


model = GPTLanguageModel()
m = model.to(device)

# 获取一个模型的参数数量
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

# 创建一个AdamW优化器，作用于model的所有参数
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# 评估模型时，我们将context设置为no_grad
@torch.no_grad()
def estimate_loss():
    out = {}
    # 设置模型进入评价模式
    model.eval()
    # 分别用train数据和val数据进行评价
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        # 评价迭代eval_iters次
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # 重新将模型设置为训练模式
    model.train()
    return out


# 进行max_iters轮次的迭代
for iter in range(max_iters):
    # 每100次迭代，对模型评价一次
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # 获取一个batch的数据
    xb, yb = get_batch("train")

    # 调用模型，得到loss
    logits, loss = model(xb, yb)
    # 清零优化器的梯度。这是因为PyTorch的优化器在每次更新参数时都会累积梯度，所以在每次更新参数之前，我们需要清零梯度。
    optimizer.zero_grad(set_to_none=True)
    # 计算损失的反向传播。这会计算出每个参数的梯度。
    loss.backward()
    # 更新优化器的参数。这会根据每个参数的梯度和学习率来更新参数的值。
    optimizer.step()

# 将参数保存到文件中
torch.save(model.state_dict(), "gpt.checkpoint")
