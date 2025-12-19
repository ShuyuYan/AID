import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils.model import *
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

new_report = ("The visualized cervical, thoracic, and abdominal aorta are well visualized with normal course of the major arteries and their branches; bilateral subclavian arteries show mild wall thickening with focal mild luminal stenosis; the proximal segments of bilateral common carotid arteries show wall thickening without significant luminal narrowing; the remaining arteries show no significant stenosis, filling defects, or dilatation, and no other abnormalities are observed. Imaging diagnosis: focal mild stenosis of bilateral subclavian arteries and wall thickening of the proximal segments of bilateral common carotid arteries.")

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
    根据注意力得分返回颜色 (红色高关注、蓝色低关注)
    """
    cmap = plt.cm.RdBu_r  # 红蓝渐变色
    # 注意：聚合后的分数可能会变化，这里动态调整vmax可以让颜色分布更合理
    norm = mcolors.Normalize(vmin=0, vmax=max(token_attention) / 8)
    rgba = cmap(norm(att_score))
    return '#{:02x}{:02x}{:02x}'.format(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))


# --- 修改核心逻辑：合并 Subwords 并聚合注意力分数 ---

merged_words = []
merged_scores = []

# 用于临时存储当前正在合并的词和分数
current_word_parts = []
current_scores = []

# 1. 重建单词并聚合注意力权重 (解决 ## 符号问题)
words = []
word_attentions = []

for token, score in zip(tokens_decoded, token_attention):
    # 跳过特殊字符
    if token in ["[CLS]", "[SEP]", "[PAD]"]:
        continue

    # 如果是 subword (以 ## 开头)，合并到前一个词
    if token.startswith("##"):
        if words:
            words[-1] += token[2:]  # 拼接文本
            word_attentions[-1] += score  # 累加权重 (表示整个词的总关注度)
    else:
        # 新词
        words.append(token)
        word_attentions.append(score)

# 2. 根据阈值进行二值化着色 (解决颜色太淡问题)
# 阈值设置：您可以根据输出结果调整这个值。
# 建议值：所有权重的平均值，或者 1/N (N为词数)，例如 0.02 - 0.05 之间
threshold = 0.008

text_colored = []
for word, score in zip(words, word_attentions):
    if score > threshold:
        # 大于阈值：红色，加粗
        color = "red"
        font_weight = "bold"
    else:
        # 小于阈值：蓝色，正常粗细
        color = "blue"
        font_weight = "normal"

    # 生成 HTML span 标签
    colored_token = f'<span style="color:{color}; font-weight:{font_weight}; margin-right: 4px;">{word}</span>'
    text_colored.append(colored_token)

# 1. 拼接文本
html_content = " ".join(text_colored)

# 2. 简单的标点符号空格处理 (保持原有逻辑)
html_content = html_content.replace(" ,", ",").replace(" .", ".").replace(" ;", ";").replace(" :", ":")

# 3. 写入文件，重点是修改 div 的 style
with open("attention_visualization.html", "w", encoding="utf-8") as f:
    f.write(f"""
    <div style='
        font-size: 16px; 
        font-family: Arial, sans-serif; 
        line-height: 1.6; 
        white-space: normal;      /* 强制允许自动换行 */
        word-wrap: break-word;    /* 允许在长单词内部换行，防止溢出 */
        overflow-wrap: break-word;/* 标准写法，同上 */
        max-width: 1000px;        /* 限制最大宽度，避免太宽难以阅读 */
        margin: 20px;             /*以此留出一些边距 */
    '>
        {html_content}
    </div>
    """)

print("HTML 可视化结果保存为: attention_visualization.html")