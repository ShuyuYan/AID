import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from transformers import AutoTokenizer, AutoModel
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from utils.TADataset import TADataset
from utils.tools import *
"""
结合Baseline和影像报告双模态数据，使用中期融合方式预测患者治疗方案
"""


class MultiModalClassifier(nn.Module):
    def __init__(self, bert_path, tab_input_dim, num_labels):
        super(MultiModalClassifier, self).__init__()

        # BERT 模块
        self.bert = AutoModel.from_pretrained(bert_path)
        self.text_fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )

        # 表格特征模块
        self.tab_fc = nn.Sequential(
            nn.Linear(tab_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64)
        )

        # 融合 + 分类
        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask, tabular_feats):
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = text_out.last_hidden_state[:, 0, :]
        text_feat = self.text_fc(cls_output)
        tab_feat = self.tab_fc(tabular_feats)
        fused = torch.cat((text_feat, tab_feat), dim=1)
        logits = self.classifier(fused)
        return logits


def evaluate_saved_model(model, model_path, test_loader, label_encoder, device):
    """
    加载保存的模型权重，并在测试集上输出分类报告
    """
    print(f"\n=== 加载模型权重: {model_path} ===")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    preds, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['text_tokens']['input_ids'].to(device)
            attention_mask = batch['text_tokens']['attention_mask'].to(device)
            tab = batch['tab'].to(device)
            label = batch['label'].to(device)

            logits = model(input_ids, attention_mask, tab)
            predictions = torch.argmax(logits, dim=1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(label.cpu().numpy())

    # 输出结果
    acc = accuracy_score(true_labels, preds)
    print(f"\n✅ 模型评估完成！Accuracy: {acc:.4f}\n")
    print(classification_report(
        true_labels,
        preds,
        target_names=[str(c) for c in label_encoder.classes_],
        zero_division=0
    ))

    return acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_path = "/home/yanshuyu/Data/AID/TAK/Bio_ClinicalBERT"
    excel_path = os.path.expanduser('~/Data/AID/all.xlsx')
    writer = SummaryWriter(log_dir="/home/yanshuyu/Data/AID/runs/bimodal_report_tab")

    batch_size = 8
    epochs = 1000
    learning_rate = 1e-4
    max_length = 384
    best_acc = 0.60

    df = pd.read_excel(excel_path, sheet_name='try')
    target_col = df.columns[-3]
    y = df[target_col].values
    X = df.select_dtypes(include=["int64", "float64"])
    X = X.drop(columns=[target_col], errors="ignore")
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pca(X, 50)
    df_text = pd.read_excel(excel_path, sheet_name='effect1')
    report_col = 'mra_report'
    report = df_text[report_col].astype(str).tolist()[:len(X)]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    num_labels = len(label_encoder.classes_)

    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    data = TADataset(df, report, X, y, tokenizer, max_length)
    train_idx, val_idx = train_test_split(
        range(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    train_data = Subset(data, train_idx)
    val_data = Subset(data, val_idx)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    model = MultiModalClassifier(
        bert_path=bert_path,
        tab_input_dim=X.shape[1],
        num_labels=num_labels
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # # 加载预训练模型
    # model_path = "/home/yanshuyu/Data/AID/TAK/best_model/bimodal_acc0.6316_695550.pt"
    # evaluate_saved_model(model, model_path, val_loader, label_encoder, device)

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch['text_tokens']['input_ids'].to(device)
            attention_mask = batch['text_tokens']['attention_mask'].to(device)
            tab = batch['tab'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, tab)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Train Loss', avg_loss, epoch)
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0
        preds, true_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['text_tokens']['input_ids'].to(device)
                attention_mask = batch['text_tokens']['attention_mask'].to(device)
                tab = batch['tab'].to(device)
                label = batch['label'].to(device)

                logits = model(input_ids, attention_mask, tab)
                predictions = torch.argmax(logits, dim=1)
                preds.extend(predictions.cpu().numpy())
                true_labels.extend(label.cpu().numpy())
                loss = criterion(logits, label)
                val_loss += loss.item()

        acc = accuracy_score(true_labels, preds)
        writer.add_scalar('Val Loss', val_loss, epoch)
        writer.add_scalar('Val Acc', acc, epoch)
        print(f"Validation Accuracy: {acc:.4f}")
        print(classification_report(
            true_labels, preds,
            target_names=[str(c) for c in label_encoder.classes_],
            zero_division=0
        ))

        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(save_dir, f"pca_report_tab_acc{acc:.4f}_epoch{epoch + 1}_{timestamp}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"模型权重已保存至: {save_path}")
