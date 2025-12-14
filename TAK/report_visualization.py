from transformers import AutoTokenizer, AutoModel
import os
import copy
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.TADataset import TADataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils.model import *

os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT").to(device)
save_path = "/home/yanshuyu/Data/AID/TAK/checkpoints/best_model.pt"


# 加载最佳权重
model.load_state_dict(torch.load(save_path))
model.eval()

new_report = (""
              "左侧锁骨下动脉及两侧椎动脉近段轻中度狭窄，中远段管腔显示可；右锁骨下动脉近段局部稍扩张。双侧肾动脉起始段狭窄。右侧头臂干、两侧颈总动脉显影可，腹主动脉局部轻度狭窄，所示胸主动脉未见明显局限性管腔狭窄或扩展，腹腔干显示可。  影像学 诊断: 高安病：左侧锁骨下动脉及两侧椎动脉近段轻度狭窄；右锁骨下动脉近段局部瘤样扩张；双侧肾动脉起始段狭窄；腹主动脉局部轻度狭窄，请结合临床。"
              "")

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
    norm = mcolors.Normalize(vmin=0, vmax=max(token_attention)/10)  # 归一化
    rgba = cmap(norm(att_score))
    return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))


filtered_tokens = []
filtered_attentions = []
for token, score in zip(tokens_decoded, token_attention):
    if token not in ["[CLS ]", "[SE P]"]:
        filtered_tokens.append(token)
        filtered_attentions.append(score)
# 可视化注意力
text_colored = []
for token, score in zip(filtered_tokens, token_attention):
    color = attention_color(score)
    colored_token = f'<span style="color:{color}">{token}</span>'
    text_colored.append(colored_token)

# 将结果拼接成一段 HTML
from IPython.core.display import display, HTML

html_content = " ".join(text_colored)
with open("attention_visualization.html", "w", encoding="utf-8") as f:
    f.write(f"<div style='font-size:16px;font-family:Arial;'>{html_content}</div>")
print("HTML 可视化结果保存为: attention_visualization.html")
