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
"""
三模态数据融合预测最终治疗方案
"""

os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_path = "/home/yanshuyu/Data/AID/TAK/Bio_ClinicalBERT"
    # bert_path = "medicalai/ClinicalBERT"
    excel_path = "/home/yanshuyu/Data/AID/all.xlsx"
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    writer = SummaryWriter(log_dir="/home/yanshuyu/Data/AID/runs/Multimodal")
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "/home/yanshuyu/Data/AID/TAK/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_excel(excel_path, sheet_name='714')
    label_col = df.columns[-1]

    num_labels = 3
    valid_label_values = [0, 1, 2]
    max_length = 384
    batch_size = 8
    num_workers = 4
    lr = 2e-4
    num_epochs = 30
    best_val_acc = 0.9
    mra_drop_prob = 0.25

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

    tab_dim = X_np.shape[1]
    model = ImageTabTextModel(
        num_labels=num_labels,
        tab_dim=tab_dim,
        bert_path=bert_path,
        img_backbone="resnet18",
        feature_dim=256,
        tab_out_dim=128
    ).to(device)

    param_groups = [
        # A. 图像编码器 (ResNet) - 最小的LR
        {'params': model.image_encoder.parameters(), 'lr': 0.01 * lr},

        # B. 文本编码器 (BERT) - 适中的LR
        {'params': model.text_encoder.bert.parameters(), 'lr': 0.1 * lr},
        {'params': model.text_encoder.proj.parameters(), 'lr': lr},  # 投影层是新层，用基准LR

        # C. 新层/融合层/分类器 - 基础LR (用于快速学习)
        {'params': model.tab_encoder.parameters(), 'lr': lr},
        {'params': model.fusion_img_img.parameters(), 'lr': lr},
        # {'params': model.fusion_img_text.parameters(), 'lr': lr},
        # {'params': model.fusion_all.parameters(), 'lr': lr},
        {'params': model.classifier.parameters(), 'lr': lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch} [Train]'):
            head = batch['head'].to(device)
            thorax = batch['thorax'].to(device)
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

            optimizer.zero_grad()
            rand_head_mask = (torch.rand(head.size(0), 1, device=device) > mra_drop_prob).float()
            rand_thorax_mask = (torch.rand(head.size(0), 1, device=device) > mra_drop_prob).float()
            head_mask = head_mask * rand_head_mask
            thorax_mask = thorax_mask * rand_thorax_mask

            logits, _ = model(head, thorax, tab, input_ids, attention_mask, head_mask, thorax_mask)  # [B, 3]
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch} [Val]'):
                head = batch['head'].to(device)
                thorax = batch['thorax'].to(device)
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

                logits, _ = model(head, thorax, tab, input_ids, attention_mask, head_mask, thorax_mask)
                loss_batch = criterion(logits, label)
                val_loss += loss_batch.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(label.cpu().tolist())

                correct += (preds == label).sum().item()
                total += label.size(0)

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

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            print("\n=========== Classification Report ===========")
            print(classification_report(all_labels, all_preds, labels=[0, 1, 2], digits=3))

            ckpt_name = f"{start_time}_epoch{epoch}_acc{val_acc:.4f}.pth"
            best_path = os.path.join(save_dir, ckpt_name)
            torch.save(model.state_dict(), best_path)

        scheduler.step(avg_val_loss)

    writer.close()
    print("Training finished. Best val acc:", best_val_acc)
