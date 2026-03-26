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

model.load_state_dict(torch.load(save_path))
model.eval()

new_report = input()
tokens = tokenizer(new_report, return_tensors="pt")
input_ids = tokens["input_ids"].to(device)
attention_mask = tokens["attention_mask"].to(device)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
    attentions = outputs.attentions

last_layer_attentions = torch.mean(attentions[-1], dim=1)[0, :, :].cpu().numpy()
token_attention = last_layer_attentions[0] / last_layer_attentions[0].sum()
tokens_decoded = tokenizer.convert_ids_to_tokens(input_ids[0])


def attention_color(att_score):
    """
    根据注意力得分返回颜色
    """
    cmap = plt.cm.rainbow_r
    norm = mcolors.Normalize(vmin=0, vmax=max(token_attention) / 10)
    rgba = cmap(norm(att_score))
    return '#{:02x}{:02x}{:02x}'.format(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))


merged_words = []
merged_scores = []

current_word_parts = []
current_scores = []

for token, score in zip(tokens_decoded, token_attention):
    if token in ["[CLS]", "[SEP]", "[PAD]"]:
        continue

    if token.startswith("##"):
        current_word_parts.append(token[2:])
        current_scores.append(score)
    else:
        if current_word_parts:
            merged_words.append("".join(current_word_parts))
            merged_scores.append(sum(current_scores) / len(current_scores))

        current_word_parts = [token]
        current_scores = [score]

if current_word_parts:
    merged_words.append("".join(current_word_parts))
    merged_scores.append(sum(current_scores) / len(current_scores))


text_colored = []
for word, score in zip(merged_words, merged_scores):
    color = attention_color(score)
    colored_token = f'<span style="color:{color}">{word}</span>'
    text_colored.append(colored_token)

html_content = " ".join(text_colored)
html_content = html_content.replace(" ,", ",").replace(" .", ".").replace(" ;", ";").replace(" :", ":")


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