import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from transformers import AutoTokenizer, AutoModel
import os
import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
"""
结合Baseline和影像报告双模态数据，使用中期融合方式预测患者治疗方案
"""


# ========== 参数设置 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_path = "/home/yanshuyu/Data/AID/TAK/Bio_ClinicalBERT"
excel_path = os.path.expanduser('~/Data/AID/all.xlsx')
batch_size = 8
dropout = 0.3
epochs = 100
learning_rate = 5e-6
max_length = 384

# ========== 数据读取 ==========
# --- 表格数据 ---
df_tab = pd.read_excel(excel_path, sheet_name='effect1')
target_col = df_tab.columns[-3]  # 标签列（倒数第三列）
y = df_tab[target_col].values

X = df_tab.select_dtypes(include=["int64", "float64"])
X = X.drop(columns=[target_col], errors="ignore")

# 填充缺失值
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# 标准化表格数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# --- 文本数据 ---
df_text = pd.read_excel(excel_path, sheet_name='effect1')
report_col = 'mra_report'
texts = df_text[report_col].astype(str).tolist()[:len(X)]

# 标签编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
num_labels = len(label_encoder.classes_)

# 统一划分
train_texts, test_texts, train_X, test_X, train_y, test_y = train_test_split(
    texts, X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== 自定义 Dataset ==========
class MultiModalDataset(Dataset):
    def __init__(self, texts, tabular_data, labels, tokenizer, max_length):
        self.texts = texts
        self.tabular_data = tabular_data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tabular_feats = torch.tensor(self.tabular_data[idx], dtype=torch.float32)

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'tabular_feats': tabular_feats,
            'label': torch.tensor(label, dtype=torch.long)
        }

# ========== Tokenizer & DataLoader ==========
tokenizer = AutoTokenizer.from_pretrained(bert_path)
train_dataset = MultiModalDataset(train_texts, train_X, train_y, tokenizer, max_length)
test_dataset = MultiModalDataset(test_texts, test_X, test_y, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ========== 定义多模态模型 ==========
class MultiModalClassifier(nn.Module):
    def __init__(self, bert_path, tab_input_dim, num_labels):
        super(MultiModalClassifier, self).__init__()

        # BERT 模块
        self.bert = AutoModel.from_pretrained(bert_path)
        self.text_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU()
        )

        # 表格特征模块
        self.tab_fc = nn.Sequential(
            nn.Linear(tab_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 融合 + 分类
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, tabular_feats):
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = text_out.last_hidden_state[:, 0, :]
        text_feat = self.text_fc(cls_output)
        tab_feat = self.tab_fc(tabular_feats)
        fused = torch.cat((text_feat, tab_feat), dim=1)
        logits = self.classifier(fused)
        return logits

# ========== 模型初始化 ==========
model = MultiModalClassifier(
    bert_path=bert_path,
    tab_input_dim=train_X.shape[1],
    num_labels=num_labels
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# ========== 训练循环 ==========
# ====== 创建保存目录 & 获取时间戳 ======
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tabular_feats = batch['tabular_feats'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, tabular_feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")

    # ===== 验证 =====
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tabular_feats = batch['tabular_feats'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask, tabular_feats)
            predictions = torch.argmax(logits, dim=1)
            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    print(classification_report(
        true_labels, preds,
        target_names=[str(c) for c in label_encoder.classes_],
        zero_division=0
    ))

    # ===== 保存权重 =====
    save_path = os.path.join(save_dir, f"acc{acc:.4f}_epoch{epoch+1}_{timestamp}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"模型权重已保存至: {save_path}")

