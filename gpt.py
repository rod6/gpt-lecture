import torch
import torch.nn as nn
from torch.nn import functional as F

# 设置超参
# Batch，指的是大模型可以被并行处理多少不相关的训练数据，这个值也决定了一个大模型可以并发处理多少请求
batch_size = 16
# Block，决定了一次推理的上下文最大的token数量
block_size = 32
# 每个token用多少维来表达
n_embd = 64
# 多头注意力机制的数量
n_head = 4
# 有多少个Block来堆叠
n_layer = 4
# dropout的工作原理是在训练过程中随机地关闭（即将输出设置为零）一部分神经网络的神经元。通过这种方式，模型在每个训练步骤中
# 都会使用不同的神经元子集，从而减少单个神经元对局部输入模式的依赖，增强了模型的泛化能力。
#
# Dropout的优点：
# - 减少过拟合：通过减少复杂的协同适应性（即神经元间过度依赖），dropout可以有效地减少模型的过拟合现象。
# - 模型鲁棒性：由于模型在训练时学会了在缺少一部分神经元的情况下进行预测，这使得模型更加健壮，对输入数据的小变动不那么敏感。
dropout = 0.2
# 用 cuda 还是 cpu？
device = "cuda" if torch.cuda.is_available() else "cpu"

# 构建词表
# input.txt中是原始训练数据
with open("./input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 为了demo容易理解，我们采用char级别的词表，这里获取原始数据中所有的字符，构建词表
chars = sorted(list(set(text)))
vocab_size = len(chars)
# 构建词表到整数的映射，这个整数就可以理解为每个词表的token id
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# 构建词表的编码器和解码器
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

torch.manual_seed(1337)


# 定义一个最简单的LLM
# 1. 模型Head定义
# nn.Module: 所有神经网络的base class，任何神经网络都应该继承自这个class
#   所有nn.Module的子类，都应该实现__init__和forward两个函数
#   __init__负责初始化神经网络的架构
#   forward表示一个前向传播，构建神经网络的运算
class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        # 定义Head的q、k、v三个矩阵，均采用nn.Linear来表达
        # nn.Linear表示的是线性变换，主要参数：
        #   - in_features: 输入数据的维度
        #   - out_features：输出数据的维度
        #   - bias：是否需要偏移
        # nn.Linear内部有一个矩阵来保存参数
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # tril矩阵
        # tensor([[ 1,  0,  0,  0],
        #         [ 1,  1,  0,  0],
        #         [ 1,  1,  1,  0],
        #         [ 1,  1,  1,  1]])
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        # dropout
        # dropout的工作原理是在训练过程中随机地关闭（即将输出设置为零）一部分神经网络的神经元。
        # 通过这种方式，模型在每个训练步骤中都会使用不同的神经元子集，从而减少单个神经元对局部输
        # 入模式的依赖，增强了模型的泛化能力。
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 获取input的shape
        # B：batch，输入数据的相互独立的数据的数量
        # T：序列的长度
        # C：每一个token的大小
        B, T, C = x.shape

        # k、q是经过key、query矩阵相乘后的结果，输出大小为B,T,head_size
        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)

        # 计算注意力score，详见PPT
        w = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        # masked self attention
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        # 做softmax
        w = F.softmax(w, dim=-1)  # (B, T, T)
        # 做dropout
        w = self.dropout(w)
        # 根据输入计算v矩阵
        v = self.value(x)  # (B,T,head_size)
        # 计算最后的out
        out = w @ v  # (B, T, T) @ (B, T, hs) -> (B, T, head_size)
        # 最后的输出是B,T,head_size
        return out


# 2. 模型MultiHeadAttention定义
class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# 3. 模型FeedFoward定义
class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# 4. 模型Block定义
class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        # 原始论文里head_size是embedding大小，mutli-head是多个embeddding大小的head
        # 不过，现在很多具体实现时，会采用多个小head：
        # head_size = n_embd // n_head
        head_size = n_embd
        # 多头注意力
        self.sa = MultiHeadAttention(n_head, head_size)
        # 一个连接层
        self.ffwd = FeedFoward(n_embd)
        # Add & Norm
        self.ln1 = nn.LayerNorm(n_embd)
        # Add & Norm
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Norm在先
        # x = x + self.sa(self.ln1(x))
        # x = x + self.ffwd(self.ln2(x))
        # Norm在后
        x = self.ln1(x + self.sa(x))
        x = self.ln2(x + self.ffwd(x))
        return x


# 5. 模型GPTLanguageModel定义
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # 为了demo，token的embedding是通过学习获取
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 为了demo，position的embedding是通过学习获取
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # 多个block进行简单的堆叠
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        # 最后一个layer norm层
        self.ln_f = nn.LayerNorm(n_embd)
        # 线性层，输出大小是词表大小
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # 对参数进行初始化
        self.apply(self._init_weights)

    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        # 如果 module 是 nn.Linear 类型（即全连接层），那么它的权重将被初始化为均值为 0，标准差为 0.02 的正态分布。
        # 如果全连接层有偏置项 (bias)，那么偏置项将被初始化为 0。
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # 如果 module 是 nn.Embedding 类型（即嵌入层），那么它的权重也将被初始化为均值为 0，标准差为 0.02 的正态分布。
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # 前向传播
    def forward(self, input, targets=None):
        # input的shape是B,T，B是batch size，T是block size
        B, T = input.shape

        # 用token_embedding_table来查找input每一个token的embedding
        tok_emb = self.token_embedding_table(input)  # (B,T,C)
        # 根据T，查找每一个位置的position embedding
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        # token embedding和position embedding相加，得到最下层的输入
        x = tok_emb + pos_emb  # (B,T,C)
        # 经过所有的block前向传播
        x = self.blocks(x)  # (B,T,C)
        # 经过最后一个layer norm
        x = self.ln_f(x)  # (B,T,C)
        # 最后一个线性层，转化为logits
        # logits指模型最后输出的结果数值，一般会接一个softmax或者sigmoid等激活函数
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # 从idx中截取最后的block列，作为condition
            idx_cond = idx[:, -block_size:]
            # 获取logits
            logits, loss = self(idx_cond)
            # 只关心最后一步的结果
            logits = logits[:, -1, :]  # becomes (B, C)
            # 做一次softmax
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # 根据probs，确定idx_next
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 将idx_next放入idx
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
