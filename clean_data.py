import re
import os

def clean_data(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"找不到文件 {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    cleaned = []
    # 匹配罗马数字（如 ⅰ, ⅱ, ⅲ 等）
    roman_pattern = re.compile(r'^[ⅰ-ⅹ\s]+$')
    # 匹配带括号的罗马数字结构描述（如 ⅰ（乾下乾上））
    roman_bracket_pattern = re.compile(r'^[ⅰ-ⅹ]+[（$].*?[）$]$')
    
    for line in lines:
        # 去除首尾的空白字符，包括全角的排版空格
        s = line.strip(' \t\n\r　')
        
        # 过滤掉空行
        if not s:
            continue
            
        # 1. 过滤掉网站水印、电子书声明等
        if '流芳阁' in s or '书籍名称' in s or 'lfglib.cn' in s:
            continue
            
        # 2. 过滤掉纯目录或版块词汇
        if s in ['周易', '上经', '下经', '系辞上', '系辞下', '说卦', '序卦', '杂卦']:
            continue
            
        # 3. 过滤掉类似 "ⅰ" 或 "ⅰ（乾下乾上）" 这种排版用的无用符号
        if roman_pattern.match(s) or roman_bracket_pattern.match(s):
            continue
            
        # 4. 优化章节标题：将 "01. 乾（卦一）" 变成 "乾（卦一）"，去掉数字前缀防止模型学习无意义的递增序号
        s = re.sub(r'^\d+\.\s*', '', s)
        
        cleaned.append(s)

    with open(output_file, 'w', encoding='utf-8') as f:
        for c in cleaned:
            f.write(c + '\n')
            
    print("====== 数据清洗完成 ======")
    print(f"原始数据行数: {len(lines)}")
    print(f"清洗后的行数: {len(cleaned)}")
    print(f"已生成纯净训练集: {output_file}")

if __name__ == '__main__':
    clean_data('fortune_data.txt', 'fortune_data_clean.txt')