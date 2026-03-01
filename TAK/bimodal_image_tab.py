import os
import copy
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tabpfn import TabPFNClassifier  # 确保安装了 tabpfn
from transformers import AutoTokenizer

# 请确保 utils.model 中包含 GateFusion 类
from utils.TADataset import TADataset
from utils.model import GateFusion

"""
双模态数据融合（图像+表格）预测最终治疗方案
使用指定的 ImageEncoder 和 TabPFNEncoder
"""


# --- 1. 定义 ImageEncoder ---
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

    def forward(self, x):
        x = self.encoder(x)  # [B, C, 1, 1]
        x = x.flatten(1)  # [B, C]
        x = self.fc(x)  # [B, out_dim]
        return x


# --- 2. 定义 TabPFNEncoder ---
class TabPFNEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=128, n_ensemble=4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # 注意：TabPFNClassifier 初始化比较耗时，且显存占用较大
        self.tabpfn = TabPFNClassifier(
            model_path='/home/yanshuyu/Data/AID/TAK/TabPFN/tabpfn-v2-classifier-v2_default.ckpt',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            n_estimators=n_ensemble
        )

        self.fitted = False
        self.projection = nn.Linear(512, out_dim)

        self.fallback_mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, out_dim),
        )

    def fit(self, X_train, y_train):
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.detach().cpu().numpy()

        self.tabpfn.fit(X_train, y_train)
        self.fitted = True
        return self

    def extract_tabpfn_features(self, x):
        if isinstance(x, torch.Tensor):
            device = x.device
            x_np = x.detach().cpu().numpy()
        else:
            device = 'cpu'
            x_np = x

        # TabPFN 的 predict_proba 输出形状是 [N, n_classes]
        # 如果要作为特征使用，这通常维度较低。
        # 如果您想用 predict_proba 作为特征：
        proba = self.tabpfn.predict_proba(x_np)
        features = torch.tensor(proba, dtype=torch.float32, device=device)
        return features

    def forward(self, x):
        if not self.fitted:
            return self.fallback_mlp(x)

        with torch.no_grad():
            tabpfn_feat = self.extract_tabpfn_features(x)

        # 动态调整 projection 层以匹配 TabPFN 的输出维度 (通常是 num_classes)
        # 注意：这会导致在推理时重新初始化层，如果 num_classes 固定，建议在 init 中定死
        if tabpfn_feat.shape[-1] != self.projection.in_features:
            # 为了避免训练中途重置参数，建议在 fit 后或 init 确定维度。
            # 这里保留原逻辑，但需注意如果 batch 间维度不变则无影响。
            if self.projection.in_features != tabpfn_feat.shape[-1]:
                self.projection = nn.Linear(
                    tabpfn_feat.shape[-1], self.out_dim
                ).to(x.device)

        projected_feat = self.projection(tabpfn_feat)
        return projected_feat


# --- 3. 定义整合模型 ImageTabModel ---
class ImageTabModel(nn.Module):
    def __init__(self, num_labels, tab_dim, img_backbone="resnet18", feature_dim=256, tab_out_dim=128):
        super(ImageTabModel, self).__init__()

        # 图像编码器 (ResNet18)
        self.image_encoder = ImageEncoder(backbone=img_backbone, out_dim=feature_dim)

        # 表格编码器 (TabPFN)
        self.tab_encoder = TabPFNEncoder(in_dim=tab_dim, out_dim=tab_out_dim)

        # 图像融合层 (GateFusion)
        # 将 Head, Thorax, Leg 三个特征 (feature_dim) 融合为一个 (feature_dim)
        self.img_gate_fusion = GateFusion(feature_dim, feature_dim, feature_dim)

        # 分类器 (图像特征 + 表格特征)
        self.classifier = nn.Linear(feature_dim + tab_out_dim, num_labels)

    def fit_tab_encoder(self, X, y):
        self.tab_encoder.fit(X, y)

    def forward(self, head, thorax, leg, tab, head_mask=None, thorax_mask=None, leg_mask=None):
        # --- 图像模态 ---
        # 提取特征 [B, feature_dim]
        h_feat = self.image_encoder(head)
        t_feat = self.image_encoder(thorax)
        l_feat = self.image_encoder(leg)

        # 应用 Mask
        if head_mask is not None: h_feat = h_feat * head_mask
        if thorax_mask is not None: t_feat = t_feat * thorax_mask
        if leg_mask is not None: l_feat = l_feat * leg_mask

        # 融合三个部位 [B, 3, feature_dim] -> [B, feature_dim]
        img_fused = self.img_gate_fusion(h_feat, t_feat)

        # --- 表格模态 ---
        tab_feat = self.tab_encoder(tab)

        # --- 最终融合与分类 ---
        combined_feat = torch.cat([img_fused, tab_feat], dim=1)
        logits = self.classifier(combined_feat)

        return logits, combined_feat


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    excel_path = "/home/yanshuyu/Data/AID/all.xlsx"
    bert_path = "/home/yanshuyu/Data/AID/TAK/Bio_ClinicalBERT"
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "/home/yanshuyu/Data/AID/TAK/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    # --- 数据加载 ---
    df = pd.read_excel(excel_path, sheet_name='in')
    label_col = df.columns[-1]

    num_labels = 3
    max_length = 384
    batch_size = 8
    num_workers = 4
    lr = 1e-4
    num_epochs = 100
    best_val_acc = 0.5
    mra_drop_prob = 0.25

    X = df.select_dtypes(include=['int64', 'float64'])
    X = X.drop(columns=[label_col, 'pred'], errors='ignore')
    imputer = SimpleImputer(strategy="mean")
    X_np = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_np)

    report = df['mra_report'].astype(str).tolist()[:len(X_np)]
    labels_series = df[label_col].astype(int)
    y_for_dataset = labels_series.values

    # Dataset 初始化 (不需要 tokenizer)
    data = TADataset(df, report, X_np, y_for_dataset, tokenizer, max_length=max_length)

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

    # --- 模型初始化 ---
    tab_dim = X_np.shape[1]
    model = ImageTabModel(
        num_labels=num_labels,
        tab_dim=tab_dim,
        img_backbone="resnet18",
        feature_dim=256,
        tab_out_dim=128
    )

    # --- 训练 TabPFN ---
    print("Fitting TabPFN on labeled training data...")
    X_tab_for_fit = X_np[train_lab_idx]
    y_for_fit = y_for_dataset[train_lab_idx]
    model.fit_tab_encoder(X_tab_for_fit, y_for_fit)

    model = model.to(device)

    # --- 优化器设置 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    # --- 训练循环 ---
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch} [Train]'):
            head = batch['head'].to(device)
            thorax = batch['thorax'].to(device)
            leg = batch['leg'].to(device)
            tab = batch['tab'].to(device).float()
            label = batch['label'].to(device)

            head_mask = batch['head_mask'].to(device)
            thorax_mask = batch['thorax_mask'].to(device)
            leg_mask = batch['leg_mask'].to(device)

            optimizer.zero_grad()

            # 随机 Mask 增强
            rand_head_mask = (torch.rand(head.size(0), 1, device=device) > mra_drop_prob).float()
            rand_thorax_mask = (torch.rand(head.size(0), 1, device=device) > mra_drop_prob).float()
            rand_leg_mask = (torch.rand(leg.size(0), 1, device=device) > mra_drop_prob).float()

            head_mask = head_mask * rand_head_mask
            thorax_mask = thorax_mask * rand_thorax_mask
            leg_mask = leg_mask * rand_leg_mask

            logits, _ = model(head, thorax, leg, tab, head_mask, thorax_mask, leg_mask)

            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- 验证循环 ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
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

                head_mask = batch['head_mask'].to(device)
                thorax_mask = batch['thorax_mask'].to(device)
                leg_mask = batch['leg_mask'].to(device)

                logits, _ = model(head, thorax, leg, tab, head_mask, thorax_mask, leg_mask)
                loss_batch = criterion(logits, label)
                val_loss += loss_batch.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(label.cpu().tolist())

                correct += (preds == label).sum().item()
                total += label.size(0)
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

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} |"
              f" Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc >= 0.7:
            best_val_acc = val_acc
            print("\n=========== Classification Report ===========")
            print(classification_report(all_labels, all_preds, labels=[0, 1, 2], digits=3))
            y_score = torch.cat(all_s, dim=0).numpy()
            np.savez('/home/yanshuyu/Data/AID/results/image_tab.npz',
                     y_score=y_score, model_name='Bnimodal (Image+SCD)')

    print("Training finished. Best val acc:", best_val_acc)
