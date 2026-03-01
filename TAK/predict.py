import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.TADataset import TADataset
from utils.model import ImageTabTextModel
from sklearn.metrics import classification_report


def set_all_seeds(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(checkpoint_path, num_labels, tab_dim, bert_path, device):
    # 加载checkpoint添加weights_only=False）
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint['model_state_dict']
    tabpfn_classifier = checkpoint.get('tabpfn_classifier', None)
    projection_in_features = checkpoint.get('projection_in_features', None)
    imputer = checkpoint.get('imputer', None)
    scaler = checkpoint.get('scaler', None)
    X_tab_context = checkpoint.get('tabpfn_X_context', None)
    y_tab_context = checkpoint.get('tabpfn_y_context', None)

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
    model.load_state_dict(state_dict)
    model.tab_encoder.fit(X_tab_context, y_tab_context)

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


def plot_confusion_matrix(cm, classes, output_path):
    plt.figure(figsize=(6, 5))
    cm = cm.astype(np.float64)
    cm = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes,
                yticklabels=classes, annot_kws={"fontsize": 18}, vmin=0.0, vmax=1.0,
                cbar_kws={"ticks": np.linspace(0, 1, 6)})
    plt.title('Confusion Matrix', fontsize=16, pad=12)
    plt.ylabel('True Label', fontsize=14, labelpad=10)
    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true, y_score, n_classes, output_path):
    plt.figure(figsize=(6, 5))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_true_onehot = np.eye(n_classes)[y_true]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f"Treatment {chr(ord('A') + i)} (AUC = {roc_auc[i]:.3f})")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, labelpad=10)
    plt.ylabel('True Positive Rate', fontsize=14, labelpad=10)
    plt.title('ROC Curve on Internal Test Set', fontsize=16, pad=12)
    # plt.title('ROC Curve on External Test Set', fontsize=16, pad=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def predict(
        excel_path: str = '/home/yanshuyu/Data/AID/all.xlsx',
        checkpoint_path: str = '/home/yanshuyu/Data/AID/TAK/checkpoints/20260204_202736TrueTrueTrue_fold2_epoch2_acc0.9355.pth',
        train_excel_path: str = '/home/yanshuyu/Data/AID/all.xlsx',
        bert_path: str = "/home/yanshuyu/Data/AID/TAK/Bio_ClinicalBERT",
        output_path: str = '/home/yanshuyu/Data/AID/out1.xlsx',
        batch_size: int = 8,
        num_workers: int = 4,
        max_length: int = 384,
        num_labels: int = 3,
        do_eval: bool = True,
        val_split: float = 0.2
        # val_split: float = 0
):
    set_all_seeds(42)
    """主预测函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    # 读取待预测数据
    df = pd.read_excel(excel_path, sheet_name='in')
    label_col = 'type'
    print(f"加载待预测数据:  {len(df)} 条记录")

    # 如果是评估模式，处理数据集划分
    val_indices = []
    if do_eval:
        if val_split > 0:
            indices = np.arange(len(df))
            try:
                labels = df[label_col].values
                _, _, _, _, train_idx, val_idx = train_test_split(
                    df, labels, indices, test_size=val_split, stratify=labels, random_state=42
                )
            except Exception as e:
                print(f"无法进行分层划分（可能缺少标签列），使用随机划分: {e}")
                train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42)

            df['dataset_split'] = 'train'
            df.iloc[val_idx, df.columns.get_loc('dataset_split')] = 'val'
            val_indices = val_idx
            print(f"测试集 {len(val_idx)} 条")
        else:
            val_indices = np.arange(len(df))
            df['dataset_split'] = 'val'
            print(f"数据集大小: {len(df)} 条 (全部用于评估)")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_imputer = None
    saved_scaler = None

    if isinstance(checkpoint, dict):
        saved_imputer = checkpoint.get('imputer', None)
        saved_scaler = checkpoint.get('scaler', None)

    # 确定预处理器
    if saved_imputer is not None and saved_scaler is not None:
        imputer = saved_imputer
        scaler = saved_scaler
    elif train_excel_path is not None:
        train_df = pd.read_excel(train_excel_path, sheet_name='in')
        train_label_col = 'type'
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
    report = df['mra_report'].astype(str).tolist()[:len(X_np)]
    # report = df['mra_report_ch'].astype(str).tolist()[:len(X_np)]

    # 准备标签
    if do_eval:
        # 评估模式下，读取真实标签
        try:
            true_labels = df[label_col].astype(int).values
        except Exception:
            true_labels = np.full(len(X_np), -1, dtype=np.int64)
    else:
        true_labels = np.full(len(X_np), -1, dtype=np.int64)

    dataset = TADataset(df, report, X_np, true_labels, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model, _, _ = load_model(checkpoint_path, num_labels, tab_dim, bert_path, device)

    all_preds = []
    all_probs = []
    all_labels = []
    all_s = []
    # all_l = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='预测中'):
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

            logits, _ = model(head, thorax, leg, tab, input_ids, attention_mask, head_mask, thorax_mask, leg_mask)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_s.append(probs.cpu())
            # all_l.append(label.cpu())

    df['predicted_label'] = all_preds
    y_score = torch.cat(all_s, dim=0).numpy()[val_indices]
    # y_true = torch.cat(all_l, dim=0).numpy()[val_indices]
    # np.save('/home/yanshuyu/Data/AID/results/y_true.npy', y_true)
    np.savez('/home/yanshuyu/Data/AID/results/multimodal.npz', y_score=y_score, model_name='Multimodal')
    val_labels = [all_labels[i] for i in val_indices]
    val_preds = [all_preds[i] for i in val_indices]
    print(classification_report(val_labels, val_preds, labels=[0, 1, 2], digits=3))
    all_probs = np.array(all_probs)
    for i in range(num_labels):
        df[f'prob_class_{i}'] = all_probs[:, i]

    label_mapping = {
        0: "Treatment A",
        1: "Treatment B",
        2: "Treatment C"
    }
    df['predicted_treatment'] = df['predicted_label'].map(label_mapping)

    if output_path is None:
        base, ext = os.path.splitext(excel_path)
        output_path = f"{base}_predicted{ext}"

    df.to_excel(output_path, index=False)
    print(f"预测完成！结果已保存至: {output_path}")

    if do_eval and len(val_indices) > 0:
        print("\n========== 验证集评估报告 ==========")
        val_true = true_labels[val_indices]
        val_pred = np.array(all_preds)[val_indices]
        val_prob = all_probs[val_indices]

        acc = np.mean(val_true == val_pred)
        print(f"验证集准确率 (Accuracy): {acc:.4f}")

        output_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]

        # 1. 绘制混淆矩阵
        cm = confusion_matrix(val_true, val_pred)
        cm_path = os.path.join(output_dir, f"{base_name}_confusion_matrix.png")
        plot_confusion_matrix(cm, list(label_mapping.values()), cm_path)

        # 2. 绘制ROC曲线
        roc_path = os.path.join(output_dir, f"{base_name}_roc_curve.png")
        plot_roc_curve(val_true, val_prob, num_labels, roc_path)

    return df


if __name__ == "__main__":
    predict()
