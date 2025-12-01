import numpy as np
import torch
import os
import pandas as pd
from transformers import AutoTokenizer
import datetime
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from utils.TADataset import TADataset
from utils.model import *


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_path = "/home/yanshuyu/Data/AID/TAK/Bio_ClinicalBERT"
    excel_path = "/home/yanshuyu/Data/AID/all.xlsx"
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.read_excel(excel_path, sheet_name='714')
    label_col = df.columns[-1]

    num_labels = 3
    max_length = 384
    batch_size = 1
    num_workers = 4

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
    train_lab_idx, val_lab_idx = train_test_split(
        labeled_indices,
        test_size=0.2,
        random_state=42,
        stratify=df.loc[labeled_indices, label_col].values
    )
    val_indices = list(val_lab_idx)
    val_subset = Subset(data, val_indices)
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

    checkpoint_path = '/home/yanshuyu/Data/AID/TAK/best_model/multimodal0951.pth'
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✅ Successfully loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
    except RuntimeError as e:
        print(f"❌ Error loading state_dict due to model mismatch: {e}")
        print("Tip: 检查 ImageTabTextModel 的初始化参数是否与保存权重时一致。")

    model.eval()
    y_true = []
    y_prob = []
    with torch.no_grad():
        for batch in val_loader:
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
            y_true.append(label.detach().cpu())
            y_prob.append(F.softmax(logits.detach().cpu(), dim=1))

    print(y_true)
    print(y_prob)
    y_true = torch.stack(y_true).squeeze().numpy()
    y_prob = torch.stack(y_prob).squeeze().numpy()
    classes = [0, 1, 2]

    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = y_true_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(8, 6))

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i],
                 label=f'Class {i} ROC curve (AUC = {roc_auc[i]:.3f})')

    # micro
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC (AUC = {roc_auc["micro"]:.3f})',
             linestyle='--', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


