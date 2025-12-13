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
from utils.model import *

os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT").to(device)



excel_path = "/home/yanshuyu/Data/AID/all.xlsx"
start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = "/home/yanshuyu/Data/AID/TAK/checkpoints"
os.makedirs(save_dir, exist_ok=True)
df = pd.read_excel(excel_path, sheet_name='714')
label_col = df.columns[-1]

num_labels = 3
max_length = 384
batch_size = 8
num_workers = 4
lr = 2e-5
num_epochs = 10

X = df.select_dtypes(include=['int64', 'float64'])
X = X.drop(columns=[label_col], errors='ignore')
imputer = SimpleImputer(strategy="mean")
X_np = imputer.fit_transform(X)
scaler = StandardScaler()
X_np = scaler.fit_transform(X_np)
report = df['mra_examination_re_des_1'].astype(str).tolist()[:len(X_np)]
labels_series = df[label_col].astype(int)

y_for_dataset = labels_series.values
data = TADataset(df, report, X_np, y_for_dataset, tokenizer, max_length)

all_indices = df.index.to_numpy()
labeled_indices = df.index[df[label_col] != -1].to_numpy()
unlabeled_indices = df.index[df[label_col] == -1].to_numpy()
train_lab_idx, val_lab_idx = train_test_split(
    labeled_indices,
    test_size=0.2,
    random_state=42,
    stratify=df.loc[labeled_indices, label_col].values
)
train_indices = list(train_lab_idx) + list(unlabeled_indices.tolist())
val_indices = list(val_lab_idx)

train_subset = Subset(data, train_indices)
val_subset = Subset(data, val_indices)
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)



optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()
best_val_loss = float('inf')
save_path = "/home/yanshuyu/Data/AID/TAK/checkpoints/best_model.pt"

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        input_ids = batch["text_tokens"]["input_ids"].to(device)
        attention_mask = batch["text_tokens"]["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]  # 假设 [CLS] 的嵌入用作分类向量
        optimizer.zero_grad()

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

    # 验证
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["text_tokens"]["input_ids"].to(device)
            attention_mask = batch["text_tokens"]["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.last_hidden_state[:, 0, :]
            loss = criterion(logits, labels)
            total_val_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

    # 保存最佳权重
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved at epoch {epoch + 1}")

# 加载最佳权重
model.load_state_dict(torch.load(save_path))
model.eval()

# 输出分类报告
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))
