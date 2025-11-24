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

        self.encoder = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(in_features, out_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


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
        self.tab_proj = nn.Linear(tab_dim, img_dim)
        self.gate_fc = nn.Sequential(
            nn.Linear(img_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, img_dim),
            nn.Sigmoid()
        )

    def forward(self, img_feat, tab_feat):
        tab_feat = self.tab_proj(tab_feat)
        x = torch.cat([img_feat, tab_feat], dim=1)
        gate = self.gate_fc(x)
        fused = gate * img_feat + (1 - gate) * tab_feat
        return fused


class ImageTabularModel(nn.Module):
    def __init__(self, num_labels, tab_dim,
                 img_backbone="resnet18",
                 img_dim=256, tab_out_dim=128,
                 fused_dim=256):
        super().__init__()

        self.image_encoder = ImageEncoder(
            backbone=img_backbone,
            out_dim=img_dim
        )

        self.tab_encoder = TabularEncoder(
            in_dim=tab_dim,
            out_dim=tab_out_dim
        )

        self.fusion = GateFusion(img_dim, tab_out_dim, fused_dim)

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

    model = ImageTabularModel(
        num_labels=num_labels,
        tab_dim=tab_dim,
        img_backbone="resnet18",
        img_dim=256,
        tab_out_dim=128,
        fused_dim=256
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # ------------------------ 新增：保存最优模型 ------------------------ #
    best_val_acc = 0
    save_path = "/home/yanshuyu/Data/AID/TAK/best_model/best_bimodal_model.pth"  # ← 新增

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]'):
            head = batch['head'].to(device)
            tab = batch['tab'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(head, tab)
            probs = torch.softmax(logits, dim=1)

            mask_valid = (label != -1)
            loss_sup = criterion(logits[mask_valid], label[mask_valid]) if mask_valid.sum() > 0 else torch.tensor(0.0)

            mask_invalid = (label == -1)
            loss_pseudo = torch.tensor(0.0, device=device)

            if mask_invalid.sum() > 0:
                invalid_probs = probs[mask_invalid]
                max_prob, pseudo_label = torch.max(invalid_probs, dim=1)

                mask_high = (max_prob >= 0.9)
                if mask_high.sum() > 0:
                    loss_pseudo += criterion(
                        logits[mask_invalid][mask_high],
                        pseudo_label[mask_high]
                    ) * 1.0

                mask_mid = (max_prob < 0.9) & (max_prob >= 0.6)
                if mask_mid.sum() > 0:
                    soft_target = invalid_probs[mask_mid].detach()
                    pred = probs[mask_invalid][mask_mid]
                    loss_kl = torch.nn.KLDivLoss(reduction="batchmean")(
                        torch.log(pred + 1e-12), soft_target
                    )
                    loss_pseudo += 0.3 * loss_kl

            loss = loss_sup + loss_pseudo
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ------------------------ Validation ------------------------ #
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]'):
                head = batch['head'].to(device)
                tab = batch['tab'].to(device)
                label = batch['label'].to(device)

                mask_valid = (label != -1)
                if mask_valid.sum() == 0:
                    continue

                logits = model(head, tab)
                loss = criterion(logits[mask_valid], label[mask_valid])
                val_loss += loss.item()

                pred = torch.argmax(logits[mask_valid], dim=1)
                all_preds.extend(pred.cpu().tolist())
                all_labels.extend(label[mask_valid].cpu().tolist())

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)

        writer.add_scalar('Train Loss', avg_train_loss, epoch)
        writer.add_scalar('Val Loss', avg_val_loss, epoch)
        writer.add_scalar('Val Acc', val_acc, epoch)

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}"
              f" | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # ------------------------ 新增：输出 classification report ------------------------ #
        print("\n=== Classification Report ===")
        print(classification_report(
            all_labels,
            all_preds,
            target_names=[str(c) for c in label_encoder.classes_],
            digits=3
        ))

        # ------------------------ 新增：保存最优模型 ------------------------ #
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"🔥 Saved best model (Acc={best_val_acc:.4f}) to {save_path}")

        scheduler.step(avg_val_loss)

    writer.close()
