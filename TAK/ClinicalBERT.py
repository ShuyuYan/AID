import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ======= 参数设置 =======
excel_path = os.path.expanduser('~/Data/AID/all.xlsx')
sheet_name = 'effect1'
report_col = 'mra_report'
label_col = 'type'
bert_path = "/home/yanshuyu/Data/AID/TAK/Bio_ClinicalBERT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_length = 384
batch_size = 16
epochs = 20
learning_rate = 2e-5

# ======= 1. 读取数据 =======
df = pd.read_excel(excel_path, sheet_name=sheet_name)
texts = df[report_col].astype(str).tolist()
labels = df[label_col].astype(str).tolist()

# 标签编码
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
num_labels = len(label_encoder.classes_)

# ======= 2. 自定义Dataset =======
class ReportDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
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
            'label': torch.tensor(label, dtype=torch.long)
        }

# ======= 3. 划分训练/测试 =======
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

tokenizer = AutoTokenizer.from_pretrained(bert_path)
train_dataset = ReportDataset(train_texts, train_labels, tokenizer, max_length)
test_dataset = ReportDataset(test_texts, test_labels, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ======= 4. 定义模型 =======
class ClinicalBERTClassifier(nn.Module):
    def __init__(self, bert_path, num_labels):
        super(ClinicalBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


model = ClinicalBERTClassifier(bert_path, num_labels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# ======= 6. 训练循环 =======
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")

    # ===== 验证集评估 =====
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    print(classification_report(true_labels, preds, target_names=label_encoder.classes_))
