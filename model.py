import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FortuneTellerConfig:
    """
    模型配置类，方便统一管理超参数
    """
    def __init__(self):
        self.vocab_size = 21128      # 词表大小
        self.d_model = 768          # 嵌入维度
        self.n_head = 12             # 注意力头数
        self.n_layer = 12           # Transformer/RetNet 层数
        self.max_seq_len = 256      # 最大序列长度
        self.dropout = 0.1          # Dropout 概率
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================
# RetNet (Retentive Network) 实现
# ==============================================================

def build_decay_mask(seq_len, n_head, device):
    """
    构建多尺度指数衰减掩码 (Multi-Scale Exponential Decay Mask)
    """
    # 根据论文: gamma = 1 - 2^(-5 - i)
    gammas = 1 - 2 ** (-5 - torch.arange(n_head, dtype=torch.float, device=device))
    
    n = torch.arange(seq_len, device=device).unsqueeze(1)
    m = torch.arange(seq_len, device=device).unsqueeze(0)
    dist = n - m
    mask = (dist >= 0).float() # 因果掩码
    
    # [n_head, seq_len, seq_len]
    decay = gammas.view(-1, 1, 1) ** dist.clamp(min=0)
    decay = decay * mask
    return decay

def get_rotary_emb(seq_len, d_head, device):
    """
    计算旋转位置编码
    """
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(seq_len, device=device, dtype=torch.float)
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)

def apply_rotary_pos_emb(x, cos, sin):
    """应用旋转位置编码"""
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    x_rot = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (x_rot * sin)

class MultiScaleRetention(nn.Module):
    """
    多尺度保留机制 (MSR, Multi-Scale Retention) 
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = self.d_model // self.n_head
        
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.group_norm = nn.GroupNorm(self.n_head, self.d_model)

    def forward(self, x):
        B, T, C = x.size()
        
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        
        cos, sin = get_rotary_emb(T, self.d_head, x.device)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        decay_mask = build_decay_mask(T, self.n_head, x.device)
        
        qk = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        qk = qk * decay_mask.unsqueeze(0)
        
        out = qk @ v
        
        out = out.transpose(1, 2).contiguous().view(B * T, C)
        out = self.group_norm(out)
        out = out.view(B, T, C)
        
        return self.out_proj(out)

class GLU(nn.Module):
    """门控线性单元"""
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 2 * config.d_model, bias=False)
        self.fc2 = nn.Linear(config.d_model, 2 * config.d_model, bias=False)
        self.fc3 = nn.Linear(2 * config.d_model, config.d_model, bias=False)
        
    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))

class RetNetBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.retention = MultiScaleRetention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffn = GLU(config)

    def forward(self, x):
        x = x + self.retention(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x

class FortuneTellerModel(nn.Module):
    """
    基于 RetNet 的模型 (被 train.py 调用)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        self.h = nn.ModuleList([RetNetBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"RetNet 模型构建完成! 参数量: {n_params/1e6:.2f}M")

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
        b, t = idx.size()
        
        tok_emb = self.wte(idx)
        x = self.drop(tok_emb)
        
        for block in self.h:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        生成函数
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx