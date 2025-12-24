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
加载多模态模型，预测外部数据
"""

os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_path = "/home/yanshuyu/Data/AID/TAK/Bio_ClinicalBERT"
    # bert_path = "medicalai/ClinicalBERT"
    excel_path = "/home/yanshuyu/Data/AID/all.xlsx"
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    df = pd.read_excel(excel_path, sheet_name='714')
    label_col = df.columns[-1]

    max_length = 384
    num_labels = 3
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
    y = labels_series.values

    dataset = TADataset(df, report, X_np, y, tokenizer, max_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
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

    model.load_state_dict(torch.load('/home/yanshuyu/Data/AID/TAK/best_model/20251215_101250_epoch10_acc0.9580.pth'))
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
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
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(label.cpu().tolist())

            correct += (preds == label).sum().item()
            total += label.size(0)

            if total > 0:
                acc = correct / total
            else:
                acc = 0.0
                avg_train_loss = 0.0
                avg_val_loss = 0.0

            print(int(preds))
        print(acc)

