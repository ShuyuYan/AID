import os
import pandas as pd
import torch.nn as nn
import torchvision.models as models
import torch
from transformers import AutoTokenizer
import datetime
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.TADataset import TADataset
import pytorch_grad_cam
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

"""
图像单模态模型训练和权重保存
"""


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
        self.encoder = nn.Sequential(*list(model.children())[:-1])  # → [B, C, 1, 1]
        self.fc = nn.Linear(in_features, out_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_dim, 3)
        )

    def forward(self, x):
        x = self.encoder(x)       # [B, C, 1, 1]
        x = x.flatten(1)          # [B, C]
        x = self.fc(x)
        x = self.classifier(x)    # [B, out_dim]
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_path = "/home/yanshuyu/Data/AID/TAK/Bio_ClinicalBERT"
    excel_path = "/home/yanshuyu/Data/AID/all.xlsx"
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "/home/yanshuyu/Data/AID/TAK/checkpoints"
    df = pd.read_excel(excel_path, sheet_name='in')
    label_col = df.columns[-1]
    writer = SummaryWriter(log_dir="/home/yanshuyu/Data/AID/runs/mra")

    num_labels = 3
    max_length = 384
    batch_size = 8
    num_workers = 4

    X = df.select_dtypes(include=['int64', 'float64'])
    X = X.drop(columns=[label_col], errors='ignore')
    imputer = SimpleImputer(strategy="mean")
    X_np = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_np)
    report = df['mra_report_ch'].astype(str).tolist()[:len(X_np)]
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

    tab_dim = X_np.shape[1]
    model = ImageEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    mra_drop_prob = 0.0

    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch} [Train]'):
            head = batch['head'].to(device)
            thorax = batch['thorax'].to(device)
            label = batch['label'].to(device)
            leg = batch['leg'].to(device)
            head_mask = batch['head_mask'].to(device)
            thorax_mask = batch['thorax_mask'].to(device)
            leg_mask = batch['leg_mask'].to(device)

            optimizer.zero_grad()
            rand_head_mask = (torch.rand(head.size(0), 1, device=device) > mra_drop_prob).float()
            rand_thorax_mask = (torch.rand(head.size(0), 1, device=device) > mra_drop_prob).float()
            rand_leg_mask = (torch.rand(head.size(0), 1, device=device) > mra_drop_prob).float()
            head_mask = head_mask * rand_head_mask
            thorax_mask = thorax_mask * rand_thorax_mask
            leg_mask = leg_mask * rand_leg_mask

            logits = model(thorax)  # [B, 3]
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        best_val_acc = 0.0
        all_preds = []
        all_labels = []
        all_s = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch} [Val]'):
                head = batch['head'].to(device)
                thorax = batch['thorax'].to(device)
                leg = batch['leg'].to(device)
                tab = batch['tab'].to(device).float()
                label = batch['label'].to(device)
                text_tokens = batch.get('text_tokens', None)
                if text_tokens is not None and text_tokens.get('input_ids', None) is not None:
                    input_ids = text_tokens['input_ids'].to(device)
                    attention_mask = text_tokens['attention_mask'].to(device)
                else:
                    input_ids = None
                    attention_mask = None
                head_mask = batch['head_mask'].to(device)
                thorax_mask = batch['thorax_mask'].to(device)
                leg_mask = batch['leg_mask'].to(device)

                logits = model(thorax)
                loss_batch = criterion(logits, label)
                val_loss += loss_batch.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(label.cpu().tolist())

                correct += (preds == label).sum().item()
                total += label.size(0)
                img = thorax.detach().cpu()
                prob = F.softmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=1)
                all_s.append(probs.cpu())

            if total > 0:
                val_acc = correct / total
                avg_train_loss = total_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
            else:
                val_acc = 0.0
                avg_train_loss = 0.0
                avg_val_loss = 0.0

            writer.add_scalar('Train Loss', avg_train_loss, epoch)
            writer.add_scalar('Val Loss', avg_val_loss, epoch)
            writer.add_scalar('Val Acc', val_acc, epoch)
            print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} |"
                  f" Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print("\n=========== Classification Report ===========")
                print(classification_report(all_labels, all_preds, labels=[0, 1, 2], digits=3))

                ckpt_name = f"resnet18_{start_time}_epoch{epoch}_acc{val_acc:.4f}.pth"
                best_path = os.path.join(save_dir, ckpt_name)
                torch.save(model.state_dict(), best_path)

                y_score = torch.cat(all_s, dim=0).numpy()
                np.savez('/home/yanshuyu/Data/AID/results/image.npz', y_score=y_score, model_name='Unimodal (Image)')

            scheduler.step(avg_val_loss)

    writer.close()
    print("Training finished. Best val acc:", best_val_acc)
