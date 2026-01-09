import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils.model import *
from IPython.core.display import display, HTML
"""
加载report模型权重，提取BERT模型的注意力权重，将模型关注点可视化
"""

os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT").to(device)
save_path = "/home/yanshuyu/Data/AID/TAK/best_model/BERT_best_model.pt"

# 加载最佳权重
model.load_state_dict(torch.load(save_path))
model.eval()

new_report = input()
# 分词并转换为张量
tokens = tokenizer(new_report, return_tensors="pt")
input_ids = tokens["input_ids"].to(device)
attention_mask = tokens["attention_mask"].to(device)

# 获取模型输出和注意力权重
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
    # 如果使用的是 BertModel，则提取注意力权重
    attentions = outputs.attentions  # 这将是各层注意力权重的列表 (layers x heads x seq_len x seq_len)

# 提取最后一层的注意力权重
# 聚合多头的注意力权重 (对 heads 进行平均)
last_layer_attentions = torch.mean(attentions[-1], dim=1)[0, :, :].cpu().numpy()
# 对每个 token 的注意力权重进行归一化（行求和）
token_attention = last_layer_attentions[0] / last_layer_attentions[0].sum()

# 映射 token id 到文字
tokens_decoded = tokenizer.convert_ids_to_tokens(input_ids[0])


# 创建词的颜色映射函数
def attention_color(att_score):
    """
    根据注意力得分返回颜色
    使用 'nipy_spectral_r' colormap 来匹配图片中的彩虹色：
    Low (Red) -> Yellow -> Green -> Blue -> High (Purple)
    """
    cmap = plt.cm.rainbow_r

    # 动态调整归一化范围
    # 这里的 vmax 控制灵敏度，除以5是为了让高关注区域更明显地进入紫色/蓝色区
    norm = mcolors.Normalize(vmin=0, vmax=max(token_attention) / 10)
    rgba = cmap(norm(att_score))
    return '#{:02x}{:02x}{:02x}'.format(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))


# --- 核心逻辑：合并 Subwords 并聚合注意力分数 ---

merged_words = []
merged_scores = []

# 用于临时存储当前正在合并的词和分数
current_word_parts = []
current_scores = []

for token, score in zip(tokens_decoded, token_attention):
    # 跳过特殊字符
    if token in ["[CLS]", "[SEP]", "[PAD]"]:
        continue

    # 如果是子词（以 ## 开头）
    if token.startswith("##"):
        current_word_parts.append(token[2:])  # 去掉 ## 后追加
        current_scores.append(score)
    else:
        # 如果之前缓存了词，先结算上一个词
        if current_word_parts:
            merged_words.append("".join(current_word_parts))
            # 对注意力分数取平均值
            merged_scores.append(sum(current_scores) / len(current_scores))

        # 开始新的词
        current_word_parts = [token]
        current_scores = [score]

# 循环结束后，别忘了结算最后一个词
if current_word_parts:
    merged_words.append("".join(current_word_parts))
    merged_scores.append(sum(current_scores) / len(current_scores))

# --- 可视化生成 ---

text_colored = []
for word, score in zip(merged_words, merged_scores):
    color = attention_color(score)
    # 生成带颜色的 HTML span
    colored_token = f'<span style="color:{color}">{word}</span>'
    text_colored.append(colored_token)

# 拼接文本
html_content = " ".join(text_colored)
# 后处理标点符号
html_content = html_content.replace(" ,", ",").replace(" .", ".").replace(" ;", ";").replace(" :", ":")

# --- 生成带有图注（Legend）的 HTML ---

# 定义 CSS 渐变色，对应 nipy_spectral_r 的颜色顺序 (大致为: 红 -> 黄 -> 绿 -> 蓝 -> 紫)
# 注意：Linear-gradient 的方向是从左到右
gradient_css = "linear-gradient(to right, #e60000, #ffcc00, #00b300, #0066cc, #800080)"

html_template = f"""
<div style='font-family: Arial, sans-serif; max-width: 1000px; margin: 20px auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>

    <!-- 顶部图注区域 -->
    <div style="margin-bottom: 25px; text-align: center;">
        <h3 style="margin: 0 0 10px 0; color: #333;">Text Attention</h3>

        <!-- 彩虹条容器 -->
        <div style="width: 300px; margin: 0 auto;">
            <!-- 彩虹渐变条 -->
            <div style="
                height: 15px; 
                width: 100%; 
                background: {gradient_css}; 
                border-radius: 3px;
            "></div>

            <!-- 文字标签 Lowest / Highest -->
            <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 12px; color: #555; font-weight: bold;">
                <span>Lowest</span>
                <span>Highest</span>
            </div>
        </div>
    </div>

    <!-- 文本内容区域 -->
    <div style='
        font-size: 16px; 
        line-height: 1.6; 
        white-space: normal;      
        word-wrap: break-word;    
        overflow-wrap: break-word;
        text-align: justify;
    '>
        {html_content}
    </div>
</div>
"""

with open("attention_visualization.html", "w", encoding="utf-8") as f:
    f.write(html_template)

print("HTML 可视化结果保存为: attention_visualization.html")