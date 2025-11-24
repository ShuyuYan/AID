import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from transformers import AutoTokenizer, AutoModel
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from utils.TADataset import TADataset


# 图像特征提取器
class ImageEncoder(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, out_dim=256):
        super().__init__()
        if backbone == "resnet18":
            model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            in_features = 512
        elif backbone == "resnet50":
            model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            in_features = 2048
        else:
            raise ValueError("Unsupported backbone")

        # 去掉最后的分类层
        self.encoder = nn.Sequential(*list(model.children())[:-1])  # → [B, C, 1, 1]
        self.fc = nn.Linear(in_features, out_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x  # shape [B, out_dim]


# 报告文本特征提取器
class ReportEncoder(nn.Module):
    def __init__(self, bert_path, out_dim=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_path)
        self.text_fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )

    def forward(self, input_ids, attention_mask):
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = text_out.last_hidden_state[:, 0, :]  # 获取[CLS] token的输出
        text_feat = self.text_fc(cls_output)
        return text_feat


# 双模态融合模型
class MultiModalClassifier(nn.Module):
    def __init__(self, bert_path, tab_input_dim, num_labels):
        super(MultiModalClassifier, self).__init__()

        # 图像模块
        self.image_encoder = ImageEncoder(backbone="resnet18", out_dim=256)

        # 报告文本模块
        self.report_encoder = ReportEncoder(bert_path=bert_path, out_dim=256)

        # 融合模块
        self.fusion_fc = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_labels)
        )

    def forward(self, image, input_ids, attention_mask):
        # 图像特征
        img_feat = self.image_encoder(image)

        # 报告文本特征
        text_feat = self.report_encoder(input_ids, attention_mask)

        # 融合特征
        fused_feat = torch.cat([img_feat, text_feat], dim=1)
        logits = self.fusion_fc(fused_feat)
        return logits


# 主程序
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_path = "/home/yanshuyu/Data/AID/TAK/Bio_ClinicalBERT"
    excel_path = "/home/yanshuyu/Data/AID/all.xlsx"
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    writer = SummaryWriter(log_dir="/home/yanshuyu/Data/AID/runs/bimodal_image_report")

    max_length = 384
    num_epochs = 200
    lr = 1e-4

    df = pd.read_excel(excel_path, sheet_name='effect1')
    label_col = df.columns[-1]
    X = df.select_dtypes(include=['int64', 'float64'])
    X = X.drop(columns=[label_col], errors='ignore')
    y = df[label_col].values
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    report = df['mra_examination_re_des_1'].astype(str).tolist()[:len(X)]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    num_labels = len(label_encoder.classes_)

    train_idx, val_idx = train_test_split(
        range(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    data = TADataset(df, report, X, y, tokenizer, max_length)
    train_data = Subset(data, train_idx)
    val_data = Subset(data, val_idx)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
    tab_dim = X.shape[1]

    model = MultiModalClassifier(
        bert_path=bert_path,
        tab_input_dim=tab_dim,
        num_labels=num_labels
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_acc = 0
    save_path = "/home/yanshuyu/Data/AID/TAK/best_model/best_bimodal_model.pth"

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]'):
            image = batch['head'].to(device)
            input_ids = batch['text_tokens']['input_ids'].to(device)
            attention_mask = batch['text_tokens']['attention_mask'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()

            logits = model(image, input_ids, attention_mask)

            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # -------- 验证 -------- #
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]'):
                image = batch['head'].to(device)
                input_ids = batch['text_tokens']['input_ids'].to(device)
                attention_mask = batch['text_tokens']['attention_mask'].to(device)
                label = batch['label'].to(device)

                logits = model(image, input_ids, attention_mask)

                loss = criterion(logits, label)
                val_loss += loss.item()

                pred = torch.argmax(logits, dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        writer.add_scalar('Train Loss', avg_train_loss, epoch)
        writer.add_scalar('Val Loss', avg_val_loss, epoch)
        writer.add_scalar('Val Acc', val_acc, epoch)

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}"
              f" | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 输出分类报告
        print("\n=== Classification Report ===")
        print(classification_report(
            all_labels,
            all_preds,
            target_names=[str(c) for c in label_encoder.classes_],
            digits=3
        ))

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"🔥 Saved best model (Acc={best_val_acc:.4f}) to {save_path}")

        scheduler.step(avg_val_loss)

    writer.close()
