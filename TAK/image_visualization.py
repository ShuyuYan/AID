import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
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
"""
ResNet梯度权重可视化，输出原始MRA叠加热图
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
        x = self.classifier(x)# [B, out_dim]
        return x


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # 注册 Hook 以获取前向传播的特征图和反向传播的梯度
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        output = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)

        self.model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward()

        grads = self.gradients.cpu().data.numpy()[0]  # shape: [C, 7, 7]
        fmaps = self.activations.cpu().data.numpy()[0]  # shape: [C, 7, 7]

        weights = np.mean(grads, axis=(1, 2))  # shape: [C]

        cam = np.zeros(fmaps.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * fmaps[i]

        cam = np.maximum(cam, 0)

        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))  # Resize to (W, H)
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)  # Normalize to [0, 1]

        return cam


def show_cam_on_image(img_tensor, mask, alpha=0.3):
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)

    img = img_tensor.cpu().detach().numpy().transpose(1, 2, 0)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    img_uint8 = np.uint8(255 * img)

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()

    if mask.ndim == 3:
        mask = mask[0]

    # 确保mask大小和图像一致
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # cv2 默认是 BGR，转为 RGB

    # 叠加
    overlay = heatmap * alpha + img_uint8 * (1 - alpha)
    overlay = np.uint8(overlay)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"{TA} {label}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='jet')
    plt.title("Attention Map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay (Red=Focus)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig('imgvis.tiff', dpi=300)
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_path = "/home/yanshuyu/Data/AID/TAK/Bio_ClinicalBERT"
    excel_path = "/home/yanshuyu/Data/AID/all.xlsx"
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "/home/yanshuyu/Data/AID/TAK/checkpoints"
    df = pd.read_excel(excel_path, sheet_name='cz411')
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
    report = df['mra_report'].astype(str).tolist()[:len(X_np)]
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
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    tab_dim = X_np.shape[1]
    model = ImageEncoder().to(device)
    model.load_state_dict(torch.load('/home/yanshuyu/Data/AID/TAK/best_model/imgvis/1323thorax.pth'))
    target_layer = model.encoder[-2]
    grad_cam = GradCAM(model, target_layer)

    for batch in tqdm(val_loader):
        head = batch[('thorax')].to(device)
        TA = batch['TA']
        label = batch['label']
        if TA in [['TA1323'], ['TA1399']]:
            mask = grad_cam(head, class_idx=None)
            show_cam_on_image(head, mask)
            input()
