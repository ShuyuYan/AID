import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.TADataset import TADataset
from utils.model import ImageTabTextModel

"""
三模态数据融合预测流程
python TAK/predict.py --input /home/yanshuyu/Data/AID/all.xlsx 
--checkpoint /home/yanshuyu/Data/AID/TAK/checkpoints/20260114_185938_epoch9_acc0.9326.pth
--train_data /home/yanshuyu/Data/AID/all.xlsx --output /home/yanshuyu/Data/AID/out1.xlsx
"""


def load_model(checkpoint_path, num_labels, tab_dim, bert_path, device):
    # 加载checkpoint添加weights_only=False）
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 判断checkpoint格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 新格式：包含完整信息
        state_dict = checkpoint['model_state_dict']
        tabpfn_classifier = checkpoint.get('tabpfn_classifier', None)
        projection_in_features = checkpoint.get('projection_in_features', None)
        imputer = checkpoint.get('imputer', None)
        scaler = checkpoint.get('scaler', None)
    else:
        # 旧格式：只有state_dict
        state_dict = checkpoint
        tabpfn_classifier = None
        projection_in_features = None
        imputer = None
        scaler = None

        # 从state_dict推断projection维度
        proj_key = 'tab_encoder.projection.weight'
        if proj_key in state_dict:
            projection_in_features = state_dict[proj_key].shape[1]

    # 创建模型
    model = ImageTabTextModel(
        num_labels=num_labels,
        tab_dim=tab_dim,
        bert_path=bert_path,
        img_backbone="resnet18",
        feature_dim=256,
        tab_out_dim=128
    )

    # 修复projection层维度
    if projection_in_features is not None:
        model.tab_encoder.projection = nn.Linear(projection_in_features, model.tab_encoder.out_dim)

    # 恢复TabPFN分类器
    if tabpfn_classifier is not None:
        model.tab_encoder.tabpfn = tabpfn_classifier
        model.tab_encoder.fitted = True
        print("已恢复TabPFN分类器状态")

    # 加载权重
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, imputer, scaler


def preprocess_data(df, label_col, fit_imputer=None, fit_scaler=None):
    """预处理表格数据"""
    X = df.select_dtypes(include=['int64', 'float64'])
    X = X.drop(columns=[label_col, 'pred'], errors='ignore')

    if fit_imputer is None:
        imputer = SimpleImputer(strategy="mean")
        X_np = imputer.fit_transform(X)
    else:
        imputer = fit_imputer
        X_np = imputer.transform(X)

    if fit_scaler is None:
        scaler = StandardScaler()
        X_np = scaler.fit_transform(X_np)
    else:
        scaler = fit_scaler
        X_np = scaler.transform(X_np)

    return X_np, imputer, scaler


def predict(
        excel_path: str,
        checkpoint_path: str,
        train_excel_path: str = None,
        bert_path: str = "/home/yanshuyu/Data/AID/TAK/Bio_ClinicalBERT",
        output_path: str = None,
        batch_size: int = 8,
        num_workers: int = 4,
        max_length: int = 384,
        num_labels: int = 3,
):
    """主预测函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    # 读取待预测数据
    df = pd.read_excel(excel_path, sheet_name='out')
    label_col = df.columns[-1]
    print(f"加载待预测数据:  {len(df)} 条记录")

    # 先尝试从checkpoint加载预处理器（添加 weights_only=False）
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_imputer = None
    saved_scaler = None

    if isinstance(checkpoint, dict):
        saved_imputer = checkpoint.get('imputer', None)
        saved_scaler = checkpoint.get('scaler', None)

    # 确定预处理器
    if saved_imputer is not None and saved_scaler is not None:
        print("使用checkpoint中保存的预处理器")
        imputer = saved_imputer
        scaler = saved_scaler
    elif train_excel_path is not None:
        print(f"从训练数据拟合预处理器:  {train_excel_path}")
        train_df = pd.read_excel(train_excel_path, sheet_name='in')
        train_label_col = train_df.columns[-1]
        train_X = train_df.select_dtypes(include=['int64', 'float64'])
        train_X = train_X.drop(columns=[train_label_col, 'pred'], errors='ignore')

        imputer = SimpleImputer(strategy="mean")
        imputer.fit(train_X)
        scaler = StandardScaler()
        scaler.fit(imputer.transform(train_X))
    else:
        raise ValueError("checkpoint中没有预处理器，必须提供--train_data参数")

    # 预处理待预测数据
    X_np, _, _ = preprocess_data(df, label_col, imputer, scaler)
    tab_dim = X_np.shape[1]

    # 准备文本数据
    report = df['mra_examination_re_des_1'].astype(str).tolist()[:len(X_np)]

    # 使用-1作为占位标签
    dummy_labels = np.full(len(X_np), -1, dtype=np.int64)

    # 创建数据集和数据加载器
    dataset = TADataset(df, report, X_np, dummy_labels, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 加载模型
    print(f"加载模型: {checkpoint_path}")
    model, _, _ = load_model(checkpoint_path, num_labels, tab_dim, bert_path, device)

    # 预测
    all_preds = []
    all_probs = []

    print("开始预测...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='预测中'):
            head = batch['head'].to(device)
            thorax = batch['thorax'].to(device)
            tab = batch['tab'].to(device).float()

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

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().numpy())

    # 将预测结果添加到DataFrame
    df['predicted_label'] = all_preds

    # 添加各类别的概率
    all_probs = np.array(all_probs)
    for i in range(num_labels):
        df[f'prob_class_{i}'] = all_probs[:, i]

    # 添加预测方案的文字描述
    label_mapping = {
        0: "治疗方案A",
        1: "治疗方案B",
        2: "治疗方案C"
    }
    df['predicted_treatment'] = df['predicted_label'].map(label_mapping)

    # 保存结果
    if output_path is None:
        base, ext = os.path.splitext(excel_path)
        output_path = f"{base}_predicted{ext}"

    df.to_excel(output_path, index=False)
    print(f"预测完成！结果已保存至: {output_path}")

    # 打印统计信息
    print("\n========== 预测结果统计 ==========")
    print(df['predicted_treatment'].value_counts())
    print(f"\n总计:  {len(df)} 条记录")

    return df


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='三模态融合治疗方案预测')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='待预测的Excel文件路径')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--train_data', '-t', type=str, default=None,
                        help='训练数据Excel路径（如果checkpoint不包含预处理器则必须提供）')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出Excel文件路径')
    parser.add_argument('--bert_path', type=str,
                        default="/home/yanshuyu/Data/AID/TAK/Bio_ClinicalBERT",
                        help='BERT模型路径')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作进程数')

    args = parser.parse_args()

    predict(
        excel_path=args.input,
        checkpoint_path=args.checkpoint,
        train_excel_path=args.train_data,
        bert_path=args.bert_path,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()