import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from transformers import AutoTokenizer, AutoModel
import os
import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from utils.TADataset import TADataset



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_path = "/home/yanshuyu/Data/AID/TAK/Bio_ClinicalBERT"
    excel_path = "/home/yanshuyu/Data/AID/all.xlsx"
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    max_length = 512

    df = pd.read_excel(excel_path, sheet_name='effect1')
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

    data = TADataset(df, report, X, y, tokenizer, max_length)
    dataloader = DataLoader(data, batch_size=1, shuffle=False)

    for batch in dataloader:
        # print(batch['tabular'])
        print(batch['report'], batch['label'])

