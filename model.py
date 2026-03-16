import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FortuneTellerConfig:
    """
    模型配置类，方便统一管理超参数
    """
    def __init__(self):
        self.vocab_size = 21128      # 词表大小 (先预设一个较小的值，后续根据数据调整)
        self.d_model = 768          # 嵌入维度
        self.n_head = 12             # 注意力头数
        self.n_layer = 6            # Transformer 层数
        self.max_seq_len = 256      # 最大序列长度 (算命不需要太长)
        self.dropout = 0.1          # Dropout 概率
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CausalSelfAttention(nn.Module):
    """
    带掩码的自注意力机制 (Masked Self-Attention)
    """
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        
        self.n_head = config.n_head
        self.d_head = config.d_model // config.n_head
        
        # 注册一个 buffer 存储下三角掩码，用于因果注意力（只能看过去，不能看未来）
        self.register_buffer("bias", torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                                     .view(1, 1, config.max_seq_len, config.max_seq_len))
        
        self.dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size() # Batch, Time(seq_len), Channel(d_model)
        
        # 计算 query, key, value
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        # 拆分多头 (B, n_head, T, d_head)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # 缩放点积注意力 (Scaled Dot-Product Attention)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # 应用掩码 (Masking) - 将未来位置设为负无穷
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v # (B, n_head, T, d_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 拼回 (B, T, C)
        
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    """
    前馈神经网络 (Feed-Forward Network)
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.d_model, 4 * config.d_model)
        self.gelu    = nn.GELU() # 激活函数
        self.c_proj  = nn.Linear(4 * config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    Transformer 块: Attention + MLP
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class FortuneTellerModel(nn.Module):
    """
    算命模型主类 (基于 GPT 架构)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model), # 词嵌入
            wpe = nn.Embedding(config.max_seq_len, config.d_model), # 位置嵌入
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # 堆叠多层 Block
            ln_f = nn.LayerNorm(config.d_model), # 最终归一化层
        ))
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False) # 输出层

        # 初始化权重
        self.apply(self._init_weights)
        
        # 打印模型参数量
        n_params = sum(p.numel() for p in self.parameters())
        print(f"模型构建完成! 参数量: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        assert t <= self.config.max_seq_len, f"Cannot forward sequence of length {t}, block size is only {self.config.max_seq_len}"
        
        # 位置编码: [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) 

        # Token embeddings + Position embeddings
        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # 通过 Transformer 块
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        
        # 计算 logits
        logits = self.lm_head(x)

        # 如果提供了目标值，计算损失
        loss = None
        if targets is not None:
            # Flatten 之后计算交叉熵损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        生成函数: 给定输入序列 idx，生成后续的 token
        """
        for _ in range(max_new_tokens):
            # 截断序列以适应最大长度
            idx_cond = idx[:, -self.config.max_seq_len:]
            
            # 前向传播
            logits, _ = self(idx_cond)
            
            # 取最后一个时间步的 logits
            logits = logits[:, -1, :]
            
            # 简单的贪婪采样 (后续可以加温度采样等)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 拼接生成的 token
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

# 简单测试代码
if __name__ == "__main__":
    conf = FortuneTellerConfig()
    model = FortuneTellerModel(conf).to(conf.device)
    
    # 模拟输入: Batch=1, SeqLen=10 的随机整数
    dummy_input = torch.randint(0, conf.vocab_size, (1, 10)).to(conf.device)
    
    print(f"\n正在 {conf.device.upper()} 上运行前向传播测试...")
    logits, loss = model(dummy_input)
    print(f"输出 Logits 形状: {logits.shape}") # 预期: [1, 10, 5000]
    
    print("\n正在测试生成功能...")
    generated = model.generate(dummy_input, max_new_tokens=5)
    print(f"生成后序列形状: {generated.shape}") # 预期: [1, 15]