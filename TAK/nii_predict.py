import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from transformers import AutoTokenizer
import torchvision.models as models
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

from utils.TADataset import TADataset   # 你的 dataset 保持不变


# ==============================
# 🔥 ImageEncoder：已整合进本文件
# ==============================
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
            raise ValueError("Unsupported backbone:", backbone)

        # 去掉原始分类头
        self.encoder = nn.Sequential(*list(model.children())[:-1])  # → shape [B, C, 1, 1]
        self.fc = nn.Linear(in_features, out_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x  # shape [B, out_dim]


# ==============================
# 🔥 单模态图像分类模型
# ==============================
class ImageClassifier(nn.Module):
    def __init__(self, img_backbone="resnet18", img_dim=256, num_labels=3):
        super().__init__()
        self.img_encoder = ImageEncoder(backbone=img_backbone, out_dim=img_dim)
        self.classifier = nn.Linear(img_dim, num_labels)

    def forward(self, head_img):
        img_feat = self.img_encoder(head_img)
        logits = self.classifier(img_feat)
        return logits


# ==============================
# 🔥 主训练程序（可直接运行）
# ==============================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======= 参数 =======
    bert_path = "/home/yanshuyu/Data/AID/TAK/Bio_ClinicalBERT"
    excel_path = "/home/yanshuyu/Data/AID/all.xlsx"
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    max_length = 384
    num_epochs = 50
    lr = 1e-4
    batch_size = 4

    # ======= 读取 Excel =======
    df = pd.read_excel(excel_path, sheet_name="effect1")

    label_col = df.columns[-1]
    labels = df[label_col].values
    report = df["mra_examination_re_des_1"].astype(str).tolist()

    # 标签编码
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    num_labels = len(label_encoder.classes_)

    # tabular 数据（不用于图像模型，只用于 dataset 对齐）
    X = df.select_dtypes(include=["int64", "float64"])
    X = X.drop(columns=[label_col], errors="ignore")
    X = SimpleImputer(strategy="mean").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    # 划分数据
    train_idx, val_idx = train_test_split(
        range(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 创建 Dataset / Loader
    data = TADataset(df, report, X, y, tokenizer, max_length)
    train_loader = DataLoader(Subset(data, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(data, val_idx), batch_size=batch_size, shuffle=False)

    # ======= 初始化模型 =======
    model = ImageClassifier(
        img_backbone="resnet18",     # ← 可改成 resnet50
        img_dim=256,
        num_labels=num_labels
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ==============================
    # 🔥 训练循环
    # ==============================
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            head = batch["head"].to(device)      # ← 单模态：只用图像
            label = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(head)
            loss = criterion(logits, label)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Train Loss: {avg_loss:.4f}")

        # ==============================
        # 🔥 验证 + classification report
        # ==============================
        model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for batch in val_loader:
                head = batch["head"].to(device)
                label = batch["label"].to(device)

                logits = model(head)
                pred = torch.argmax(logits, dim=1)

                preds.extend(pred.cpu().numpy())
                trues.extend(label.cpu().numpy())

        acc = accuracy_score(trues, preds)
        print(f"Epoch {epoch + 1} Val ACC: {acc:.4f}")

        # 关键修复：把 target_names 转字符串
        target_names = [str(c) for c in label_encoder.classes_]

        print(classification_report(
            trues,
            preds,
            target_names=target_names,
            digits=3
        ))
        save_path = "/home/yanshuyu/Data/AID/TAK/best_model/"
        sp = save_path + str(epoch) + '.pt'
        torch.save(model.state_dict(), sp)

