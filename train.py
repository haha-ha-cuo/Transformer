import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import os
from model import FortuneTellerModel, FortuneTellerConfig

# ==========================================
# 1. 配置与准备
# ==========================================
class TrainConfig:
    data_path = 'fortune_data.txt'
    batch_size = 2      # 演示用，设小一点
    lr = 3e-4
    epochs = 50         # 训练轮数
    max_seq_len = 32    # 演示用，设短一点
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 2. 数据处理 (Dataset)
# ==========================================
class FortuneDataset(Dataset):
    def __init__(self, text_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # 读取并简单的按行分割数据 (实际项目中可能需要滑动窗口)
        with open(text_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        self.data = lines
        print(f"加载了 {len(self.data)} 条算命数据")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        
        # 使用 BERT 分词器编码
        # add_special_tokens=True 会自动加上 [CLS] 和 [SEP]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # input_ids: [101, 23, 45, ..., 102, 0, 0]
        input_ids = encoding['input_ids'].squeeze()
        
        # 对于 GPT 这种生成式模型：
        # 输入 (x) 是: [A, B, C, D]
        # 目标 (y) 是: [B, C, D, E] (预测下一个字)
        # 这里为了简化，我们直接用 input_ids 作为 x，y 则是 x 向后移一位
        
        x = input_ids[:-1]
        y = input_ids[1:]
        
        return x, y

# ==========================================
# 3. 训练函数
# ==========================================
def train():
    print(f"正在使用设备: {TrainConfig.device}")
    
    # A. 初始化 Tokenizer (使用 bert-base-chinese)
    # 第一次运行会自动下载词表
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    except:
        print("正在下载 bert-base-chinese 词表...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # B. 准备数据加载器
    dataset = FortuneDataset(TrainConfig.data_path, tokenizer, TrainConfig.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=TrainConfig.batch_size, shuffle=True)

    # C. 初始化模型
    model_config = FortuneTellerConfig()
    model_config.vocab_size = tokenizer.vocab_size # 同步词表大小 (21128)
    model_config.max_seq_len = TrainConfig.max_seq_len
    model_config.d_model = 256 # 演示用，改小一点
    model_config.n_layer = 4
    model_config.n_head = 4
    
    model = FortuneTellerModel(model_config).to(TrainConfig.device)
    model.train()

    # D. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainConfig.lr)

    # E. 开始训练循环
    print("\n开始训练...")
    for epoch in range(TrainConfig.epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(TrainConfig.device), y.to(TrainConfig.device)
            
            # 1. 前向传播
            logits, loss = model(x, y)
            
            # 2. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{TrainConfig.epochs} | Loss: {avg_loss:.4f}")

    print("训练完成！")
    
    # ==========================================
    # 4. 预测 / 算命演示
    # ==========================================
    print("\n[算命演示] 让模型接着说...")
    model.eval()
    
    start_text = "甲木"
    print(f"输入提示: {start_text}")
    
    # 编码输入
    input_ids = tokenizer.encode(start_text, return_tensors='pt').to(TrainConfig.device)
    # 去掉最后的 [SEP] (102)，因为我们要接着生成
    if input_ids[0, -1] == 102:
        input_ids = input_ids[:, :-1]

    # 生成
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=20)
    
    # 解码
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"模型生成: {generated_text}")

    # 保存模型
    torch.save(model.state_dict(), 'fortune_model.pth')
    print("\n模型已保存为 fortune_model.pth")

if __name__ == "__main__":
    if not os.path.exists(TrainConfig.data_path):
        print(f"请先创建 {TrainConfig.data_path}")
    else:
        train()