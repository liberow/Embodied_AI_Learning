#####################
# MiniGPT
# Author: Liberow
# Description:
#   - MiniGPT 是一个基于 PyTorch 框架从零实现的自回归生成大语言模型，旨在模仿 GPT 的核心功能。
#   - 该项目使用莎士比亚文集（1.1MB）作为训练数据集，通过预训练的方式，模型使用一张NVIDIA GeForce RTX 3090在 5 分钟内完成了训练，最终模型参数量为 0.21 M。
#   - MiniGPT 能够生成连贯的莎士比亚风格的文本。
#####################

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import logging

#################### 日志配置 ####################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("MiniGPT")

#################### 配置 ####################
@dataclass
class Config:
    batch_size: int = 16        # 每个训练批次的序列数量（并行处理序列数量）
    block_size: int = 32        # 模型上下文窗口大小，最多看到多少个 token
    max_steps: int = 5000       # 最大训练迭代次数
    eval_interval: int = 100    # 每隔多少步评估一次训练和验证集损失
    learning_rate: float = 1e-3 # AdamW 优化器的学习率
    eval_steps: int = 200       # 在评估验证集时，采样多少个小批量用于计算平均损失
    n_embd: int = 64            # token embedding 维度
    n_head: int = 4             # 多头注意力头数
    n_layer: int = 4            # Transformer 块数（层数）
    dropout: float = 0.0        # Dropout 概率，用于防止过拟合


#################### 模型模块 ####################
class FeedForward(nn.Module):
    """前馈网络：线性层 + 非线性层"""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionHead(nn.Module):
    """单个自注意力头"""

    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** (-0.5) # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # 加权聚合值
        v = self.value(x) # (B, T, C)
        out = wei @ v     # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, num_heads: int, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(head_size, n_embd, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将所有头拼接
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, n_head*C_head)
        out = self.dropout(self.proj(out))
        return out


class TransformerBlock(nn.Module):
    """Transformer 块：自注意力 + 前馈网络"""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ff = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """自回归生成大语言模型"""

    def __init__(self, vocab_size: int, config: Config, device: str):
        super().__init__()
        self.device = device
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config.n_embd, config.n_head, config.block_size, config.dropout) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                   # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=self.device)) # (T, C)
        x = tok_emb + pos_emb                                 # (B, T, C)
        x = self.blocks(x)                                    # (B, T, C)
        x = self.ln_f(x)                                      # (B, T, C)
        logits = self.lm_head(x)                              # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)                 # (B*T, C)
            targets_flat = targets.view(B*T)                  # (B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int) -> torch.Tensor:
        """根据上下文生成新token"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx


#################### 数据与训练 ####################
def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成一批训练或验证样本"""
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: Config,
    device: str,
) -> dict[str, float]:
    """评估训练和验证损失"""
    model.eval()

    def run(split: str) -> float:
        losses = []
        for _ in range(config.eval_steps):
            X, Y = get_batch(split, train_data, val_data, config.block_size, config.batch_size, device)
            _, loss = model(X, Y)
            losses.append(loss.item())
        return torch.tensor(losses).mean().item()

    results = {split: run(split) for split in ["train", "val"]}
    model.train()
    return results


def train(
    model: MiniGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: Config,
    device: str,
):
    """训练"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    model = model.to(device)

    for step in range(config.max_steps):
        if step % config.eval_interval == 0 or step == config.max_steps - 1:
            losses = evaluate_loss(model, train_data, val_data, config, device)
            logger.info(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train", train_data, val_data, config.block_size, config.batch_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model


#################### 主函数 ####################
def main():
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("-------- Hyperparameters --------")
    for k, v in vars(config).items():
        logger.info(f"{k:15s}: {v}")
    
    logger.info(f"device         : {device}")

    # 读取数据
    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # 构建词表
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    logger.info("-------- Vocabulary --------")
    logger.info(f"vocab_size     : {vocab_size}")
    logger.info(f"characters     : {chars}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    # 划分训练集与验证集
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    # 定义模型
    model = MiniGPT(vocab_size, config, device)
    logger.info("-------- Model Info --------")
    logger.info(f"parameters     : {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # 训练
    logger.info("-------- Training --------")
    model = train(model, train_data, val_data, config, device)

    # 生成
    logger.info("-------- Text Generation --------")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    output = model.generate(context, max_new_tokens=2000, block_size=config.block_size)
    logger.info(decode(output[0].tolist()))


if __name__ == "__main__":
    main()
