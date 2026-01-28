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
from sklearn.model_selection import train_test_split, StratifiedKFold
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
修改：
1. 引入五折交叉验证 + 独立测试集
2. 保持 checkpoint 字典结构与原代码完全一致，确保推理代码无需修改
"""
def create_zeros_like(tensor):
    """生成与输入 tensor 形状相同的零向量"""
    if tensor is None:
        return None
    return torch.zeros_like(tensor)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_path = "/home/yanshuyu/Data/AID/TAK/Bio_ClinicalBERT"
    excel_path = "/home/yanshuyu/Data/AID/all.xlsx"

    save_dir_base = "/home/yanshuyu/Data/AID/TAK/checkpoints"
    log_dir_base = "/home/yanshuyu/Data/AID/runs/Multimodal"
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir_base, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    df = pd.read_excel(excel_path, sheet_name='in')
    label_col = df.columns[-1]

    use_image = True
    # use_image = False
    use_report = True
    # use_report = False
    use_tabular = True
    # use_tabular = False

    num_labels = 3
    valid_label_values = [0, 1, 2]
    max_length = 384
    batch_size = 8
    num_workers = 4
    lr = 1e-3
    num_epochs = 10
    mra_drop_prob = 0.8
    n_splits = 5

    X = df.select_dtypes(include=['int64', 'float64'])
    X = X.drop(columns=[label_col, 'pred'], errors='ignore')
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

    # 1. 划分独立测试集 (20%)
    train_val_lab_idx, test_lab_idx = train_test_split(
        labeled_indices,
        test_size=0.2,
        random_state=42,
        stratify=df.loc[labeled_indices, label_col].values
    )

    test_indices = list(test_lab_idx)
    test_subset = Subset(data, test_indices)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # 2. 五折交叉验证
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_train_val = df.loc[train_val_lab_idx, label_col].values

    for fold, (train_idx_rel, val_idx_rel) in enumerate(skf.split(train_val_lab_idx, y_train_val)):
        print(f"\n{'=' * 20} Fold {fold + 1}/{n_splits} {'=' * 20}")
        best_val_acc = 0.0

        fold_train_lab_idx = train_val_lab_idx[train_idx_rel]
        fold_val_lab_idx = train_val_lab_idx[val_idx_rel]

        train_indices = list(fold_train_lab_idx) + list(unlabeled_indices.tolist())
        val_indices = list(fold_val_lab_idx)

        train_subset = Subset(data, train_indices)
        val_subset = Subset(data, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

        writer = SummaryWriter(log_dir=f"{log_dir_base}/{start_time}_fold{fold + 1}")

        tab_dim = X_np.shape[1]
        model = ImageTabTextModel(
            num_labels=num_labels,
            tab_dim=tab_dim,
            bert_path=bert_path,
            img_backbone="resnet18",
            feature_dim=256,
            tab_out_dim=128
        )

        X_tab_for_fit = X_np[fold_train_lab_idx]
        y_for_fit = y_for_dataset[fold_train_lab_idx]
        model.fit_tab_encoder(X_tab_for_fit, y_for_fit)

        model = model.to(device)

        param_groups = [
            {'params': model.image_encoder.parameters(), 'lr': 0.01 * lr},
            {'params': model.text_encoder.bert.parameters(), 'lr': 0.1 * lr},
            {'params': model.text_encoder.proj.parameters(), 'lr': lr},
            {'params': model.tab_encoder.parameters(), 'lr': lr},
            {'params': model.fusion_img_img.parameters(), 'lr': lr},
            {'params': model.classifier.parameters(), 'lr': lr},
        ]
        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        # 使用分层学习率
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

        for epoch in range(1, num_epochs + 1):
            model.train()
            total_loss = 0.0

            for batch in tqdm(train_loader, desc=f'Fold {fold + 1} Epoch {epoch} [Train]'):
                head = batch['head'].to(device) if use_image else create_zeros_like(batch['head']).to(device)
                thorax = batch['thorax'].to(device) if use_image else create_zeros_like(batch['thorax']).to(device)
                leg = batch['leg'].to(device) if use_image else create_zeros_like(batch['leg']).to(device)  # 新增 leg
                tab = batch['tab'].to(device).float() if use_tabular else create_zeros_like(batch['tab']).to(
                    device).float()
                label = batch['label'].to(device)
                text_tokens = batch.get('text_tokens', None)
                if use_report and text_tokens is not None and text_tokens.get('input_ids', None) is not None:
                    input_ids = text_tokens['input_ids'].to(device)
                    attention_mask = text_tokens['attention_mask'].to(device)
                else:
                    input_ids = None
                    attention_mask = None

                optimizer.zero_grad()
                rand_head_mask = (torch.rand(head.size(0), 1, device=device) > mra_drop_prob).float()
                rand_thorax_mask = (torch.rand(head.size(0), 1, device=device) > mra_drop_prob).float()
                rand_leg_mask = (torch.rand(leg.size(0), 1, device=device) > mra_drop_prob).float()  # 新增随机mask
                head_mask = batch['head_mask'].to(device) * rand_head_mask if use_image else create_zeros_like(
                    batch['head_mask']).to(device)
                thorax_mask = batch['thorax_mask'].to(device) * rand_thorax_mask if use_image else create_zeros_like(
                    batch['thorax_mask']).to(device)
                leg_mask = batch['leg_mask'].to(device) * rand_leg_mask if use_image else create_zeros_like(
                    batch['leg_mask']).to(device)

                logits, _ = model(head, thorax, leg, tab, input_ids, attention_mask, head_mask, thorax_mask, leg_mask)
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
                for batch in tqdm(val_loader, desc=f'Fold {fold + 1} Epoch {epoch} [Val]'):
                    head = batch['head'].to(device) if use_image else create_zeros_like(batch['head']).to(device)
                    thorax = batch['thorax'].to(device) if use_image else create_zeros_like(batch['thorax']).to(device)
                    leg = batch['leg'].to(device) if use_image else create_zeros_like(batch['leg']).to(device)  # 新增 leg
                    tab = batch['tab'].to(device).float() if use_tabular else create_zeros_like(batch['tab']).to(
                        device).float()
                    label = batch['label'].to(device)
                    text_tokens = batch.get('text_tokens', None)
                    if use_report and text_tokens is not None and text_tokens.get('input_ids', None) is not None:
                        input_ids = text_tokens['input_ids'].to(device)
                        attention_mask = text_tokens['attention_mask'].to(device)
                    else:
                        input_ids = None
                        attention_mask = None
                    head_mask = batch['head_mask'].to(device) if use_image else create_zeros_like(
                        batch['head_mask']).to(device)
                    thorax_mask = batch['thorax_mask'].to(device) if use_image else create_zeros_like(
                        batch['thorax_mask']).to(device)
                    leg_mask = batch['leg_mask'].to(device) if use_image else create_zeros_like(batch['leg_mask']).to(
                        device)

                    logits, _ = model(head, thorax, leg, tab, input_ids, attention_mask, head_mask, thorax_mask,
                                      leg_mask)
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
            print(
                f"Fold {fold + 1} Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                print(f"=== Fold {fold + 1} Best Model Updated ===")
                print(classification_report(all_labels, all_preds, labels=[0, 1, 2], digits=3))

                # 文件名中保留 fold 信息，防止覆盖
                ckpt_name = (f"{start_time}{use_image}{use_report}{use_tabular}_fold{fold + 1}"
                             f"_epoch{epoch}_acc{val_acc:.4f}.pth")
                best_path = os.path.join(save_dir_base, ckpt_name)

                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'tabpfn_X_context': X_tab_for_fit,
                    'tabpfn_y_context': y_for_fit,
                    'tabpfn_train_idx': fold_train_lab_idx,  # 更新为当前折的训练索引
                    'projection_in_features': model.tab_encoder.projection.in_features,
                    'imputer': imputer,
                    'scaler': scaler,
                }
                torch.save(checkpoint, best_path)

            scheduler.step(avg_val_loss)

        writer.close()
