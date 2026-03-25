import torch
from transformers import BertTokenizer
from model import FortuneTellerModel, FortuneTellerConfig
import os

class FortuneTellerInference:
    def __init__(self, model_path='fortune_model.pth'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"正在加载模型，使用设备: {self.device}")
        
        # 1. 加载分词器
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        # 2. 初始化模型配置 (必须与训练时的配置完全一致)
        self.config = FortuneTellerConfig()
        self.config.vocab_size = self.tokenizer.vocab_size
        self.config.max_seq_len = 128  # 与训练时保持一致
        self.config.d_model = 768
        self.config.n_layer = 12
        self.config.n_head = 12        # 与训练时保持一致
        
        # 3. 实例化模型并加载权重
        self.model = FortuneTellerModel(self.config).to(self.device)
        
        if os.path.exists(model_path):
            # 加载权重，map_location 确保在没有GPU的机器上也能用CPU加载
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"成功加载模型权重: {model_path}")
        else:
            print(f"警告: 找不到模型权重文件 {model_path}，将使用随机初始化的模型！")
            
        self.model.eval() # 切换到评估模式，关闭 dropout 等

    def predict(self, prompt_text, max_new_tokens=50):
        """
        根据提示文本生成后续内容
        """
        # 编码输入
        input_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(self.device)
        
        # 去掉最后的 [SEP] (102)，因为我们要接着这句话生成
        if input_ids[0, -1] == 102:
            input_ids = input_ids[:, :-1]

        # 如果输入太长，截断到模型能接受的最大长度
        if input_ids.size(1) > self.config.max_seq_len:
            input_ids = input_ids[:, -self.config.max_seq_len:]

        # 生成
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens)
        
        # 解码并返回
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # 去除文本中的空格 (因为 bert tokenizer 解码中文时会在字之间加空格)
        generated_text = generated_text.replace(" ", "")
        
        return generated_text

if __name__ == "__main__":
    # 初始化推理类
    teller = FortuneTellerInference(model_path='fortune_model.pth')
    
    print("\n" + "="*40)
    print("🔮 周易算命/续写模型已启动 🔮")
    print("输入 'quit' 退出")
    print("="*40 + "\n")
    
    # 交互式对话循环
    while True:
        user_input = input("请输入提示词 (例如: 《象》曰、九二、乾)：")
        if user_input.lower() == 'quit':
            break
        if not user_input.strip():
            continue
            
        print("\n模型正在推演中...")
        result = teller.predict(user_input, max_new_tokens=30)
        
        print("\n[生成结果]:")
        print(result)
        print("-" * 40 + "\n")