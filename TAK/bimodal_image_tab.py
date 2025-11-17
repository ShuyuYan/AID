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


class TabularEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),

            nn.Linear(128, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.mlp(x)


class GateFusion(nn.Module):
    def __init__(self, img_dim=256, tab_dim=128, hidden_dim=256):
        super().__init__()
        # 将表格特征升维到与图像一致
        self.tab_proj = nn.Linear(tab_dim, img_dim)

        # 门控网络，输出维度 = img_dim
        self.gate_fc = nn.Sequential(
            nn.Linear(img_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, img_dim),
            nn.Sigmoid()
        )

    def forward(self, img_feat, tab_feat):
        # tab_feat: [B, 128] → [B, 256]
        tab_feat = self.tab_proj(tab_feat)

        # 拼接后做 gating
        x = torch.cat([img_feat, tab_feat], dim=1)  # [B, 512]
        gate = self.gate_fc(x)                     # [B, 256]

        # 融合
        fused = gate * img_feat + (1 - gate) * tab_feat
        return fused


class ImageTabularModel(nn.Module):
    def __init__(self, num_labels, tab_dim,
                 img_backbone="resnet18",
                 img_dim=256, tab_out_dim=128,
                 fused_dim=256):
        super().__init__()

        # 1. 图像特征
        self.image_encoder = ImageEncoder(
            backbone=img_backbone,
            out_dim=img_dim
        )

        # 2. 表格特征
        self.tab_encoder = TabularEncoder(
            in_dim=tab_dim,
            out_dim=tab_out_dim
        )

        # 3. 融合层（核心）
        self.fusion = GateFusion(img_dim, tab_out_dim, fused_dim)

        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels)
        )

    def forward(self, image, tab):
        img_feat = self.image_encoder(image)
        tab_feat = self.tab_encoder(tab)
        fused = self.fusion(img_feat, tab_feat)
        logits = self.classifier(fused)
        return logits



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_path = "/home/yanshuyu/Data/AID/TAK/Bio_ClinicalBERT"
    excel_path = "/home/yanshuyu/Data/AID/all.xlsx"
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    writer = SummaryWriter(log_dir="/home/yanshuyu/Data/AID/runs/bimodal_image_tab")

    max_length = 384
    num_epochs = 20
    lr = 1e-4

    df = pd.read_excel(excel_path, sheet_name='effect-1')
    label_col = df.columns[-3]
    X = df.select_dtypes(include=['int64', 'float64'])
    X = X.drop(columns=[label_col], errors='ignore')
    y = df[label_col].values
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    report = df['mra_report'].astype(str).tolist()[:len(X)]
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
    tab_dim = X.shape[1]  # 表格特征维度

    model = ImageTabularModel(
        num_labels=num_labels,
        tab_dim=tab_dim,
        img_backbone="resnet18",  # 或 resnet50
        img_dim=256,  # 图像特征维度
        tab_out_dim=128,  # 表格特征维度
        fused_dim=256
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]'):
            head = batch['head'].to(device)
            tab = batch['tab'].to(device)
            label = batch['label'].to(device)  # 有效：0/1/2，无效：-1

            optimizer.zero_grad()

            # -------- Step 1: forward -------- #
            logits = model(head, tab)  # shape [B, num_labels]
            probs = torch.softmax(logits, dim=1)  # soft probabilities

            # -------- Step 2: supervised loss (有效标签) -------- #
            mask_valid = (label != -1)
            if mask_valid.sum() > 0:
                loss_sup = criterion(
                    logits[mask_valid],
                    label[mask_valid]
                )
            else:
                loss_sup = torch.tensor(0.0, device=device)

            # -------- Step 3: pseudo label for invalid samples -------- #
            mask_invalid = (label == -1)
            loss_pseudo = torch.tensor(0.0, device=device)
            loss_consistency = torch.tensor(0.0, device=device)

            if mask_invalid.sum() > 0:
                invalid_probs = probs[mask_invalid]  # [K, num_labels]
                max_prob, pseudo_label = torch.max(invalid_probs, dim=1)

                # ---- high confidence (≥0.9): use as hard label ---- #
                mask_high = (max_prob >= 0.9)
                if mask_high.sum() > 0:
                    loss_pseudo += criterion(
                        logits[mask_invalid][mask_high],
                        pseudo_label[mask_high]
                    ) * 1.0  # full weight

                # ---- medium confidence (0.6~0.9): use soft label KL ---- #
                mask_mid = (max_prob < 0.9) & (max_prob >= 0.6)
                if mask_mid.sum() > 0:
                    soft_target = invalid_probs[mask_mid].detach()
                    pred = probs[mask_invalid][mask_mid]
                    loss_kl = torch.nn.KLDivLoss(reduction="batchmean")(
                        torch.log(pred + 1e-12),
                        soft_target
                    )
                    loss_pseudo += 0.3 * loss_kl  # lower weight

                # ---- consistency loss: predict strong vs weak aug ---- #
                # 需要 strong augmented head/tab，你的 dataset 若未实现，我可帮你补
                # 这里仅示例，可先不启用
                # logits_aug = model(head_aug, tab_aug)
                # probs_aug = torch.softmax(logits_aug, dim=1)
                # loss_consistency = 0.1 * KL(probs, probs_aug)

            # -------- Step 4: total loss -------- #
            loss = loss_sup + loss_pseudo + loss_consistency
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # -------- Validation -------- #
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]'):
                head = batch['head'].to(device)
                tab = batch['tab'].to(device)
                label = batch['label'].to(device)

                # 只对有效标签做验证
                mask_valid = (label != -1)
                if mask_valid.sum() == 0:
                    continue

                logits = model(head, tab)
                loss = criterion(logits[mask_valid], label[mask_valid])
                val_loss += loss.item()

                pred = torch.argmax(logits[mask_valid], dim=1)
                correct += (pred == label[mask_valid]).sum().item()
                total += mask_valid.sum().item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        writer.add_scalar('Train Loss', avg_train_loss, epoch)
        writer.add_scalar('Val Loss', avg_val_loss, epoch)
        writer.add_scalar('Val Acc', val_acc, epoch)

        print(
            f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step(avg_val_loss)
    writer.close()

