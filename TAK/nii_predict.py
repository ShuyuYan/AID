# train_vit_offline.py
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import nibabel as nib
import torchvision.transforms.functional as F_tv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import timm
from safetensors.torch import load_file  # 用于加载本地 safetensors 权重

# -------------------------
# 可配置参数
# -------------------------
RANDOM_SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 300
LR = 1e-4
WEIGHT_DECAY = 1e-4
MODEL_NAME = "vit_base_patch16_224"
IMAGE_SIZE = (224, 224)     # (H, W)
NUM_SLICES = 3              # 你数据集中目标通道数
EXCEL_PATH = "/home/yanshuyu/Data/AID/all.xlsx"
SHEET_NAME = "effect1"
PRETRAIN_PATH = "/home/yanshuyu/Data/AID/TAK/model.safetensors"  # 离线权重路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_BEST_MODEL = "best_vit_offline.pth"

# -------------------------
# 固定随机种子
# -------------------------
def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

# -------------------------
# Dataset
# -------------------------
class NiiDataset(Dataset):
    def __init__(self, df, transform=True, target_size=(224, 224, 3), augment=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        nii_path = str(row["head"])
        label = int(row["type"])

        # 加载nii文件
        img = nib.load(nii_path).get_fdata()
        img = np.nan_to_num(img).astype(np.float32)

        # 处理3D到 num_slices 通道
        if img.ndim == 3:
            n = img.shape[2]
            needed = self.target_size[2]
            if n >= needed:
                img = img[:, :, :needed]
            else:
                pad = np.zeros((*img.shape[:2], needed), dtype=img.dtype)
                pad[:, :, :n] = img
                img = pad
        else:
            img = np.expand_dims(img, 2)
            img = np.repeat(img, self.target_size[2], axis=2)

        # resize 每层
        imgs = []
        for i in range(img.shape[2]):
            slice_img = torch.tensor(img[:, :, i])
            slice_img = TF.resize(slice_img.unsqueeze(0), self.target_size[:2])
            imgs.append(slice_img)
        img = torch.stack(imgs, dim=0).squeeze(1)


        # 归一化
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        # 数据增强
        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
            if random.random() < 0.5:
                img = TF.vflip(img)
            if random.random() < 0.5:
                angle = random.uniform(-15, 15)
                img = TF.rotate(img, angle)
            if random.random() < 0.5:
                img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
            if random.random() < 0.3:
                img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
            if random.random() < 0.5:
                img = img + torch.randn_like(img) * 0.05  # 加高斯噪声

        img = img.float()
        return img, torch.tensor(label, dtype=torch.long)

# -------------------------
# 加载数据表
# -------------------------
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
df = df[df["head"].apply(lambda x: os.path.exists(str(x)))]
df = df.reset_index(drop=True)
le = LabelEncoder()
df["type"] = le.fit_transform(df["type"].astype(str))
train_df, val_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["type"])

# 类别权重
class_counts = train_df["type"].value_counts().sort_index()
weights = 1.0 / class_counts
class_weights = torch.tensor(weights.values, dtype=torch.float32)

print("Classes:", list(le.classes_))
print("Class counts (train):", class_counts.to_dict())
print("Computed class weights:", class_weights.tolist())

# -------------------------
# DataLoader
# -------------------------
train_ds = NiiDataset(train_df, augment=True, target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_SLICES))
val_ds = NiiDataset(val_df, augment=False, target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_SLICES))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# -------------------------
# Model: timm ViT (offline)
# -------------------------
num_classes = len(le.classes_)
print(f"\nCreating ViT model ({MODEL_NAME}) with {num_classes} output classes ...")
model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=num_classes,
    drop_rate=0.5,        # 分类头 dropout
    drop_path_rate=0.2,   # transformer block 级别随机丢弃
)


# 加载离线预训练权重
print(f"Loading pretrained weights from {PRETRAIN_PATH} ...")
state_dict = load_file(PRETRAIN_PATH)

# 删除分类头的参数（ImageNet的1000类头）
for key in ["head.weight", "head.bias"]:
    if key in state_dict:
        del state_dict[key]

# 加载剩下的 backbone 权重
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("✅ Pretrained backbone loaded (classification head skipped).")
if missing:
    print("Missing keys (first few):", missing[:5])
if unexpected:
    print("Unexpected keys (first few):", unexpected[:5])

model.to(DEVICE)

# -------------------------
# Loss, Optimizer, Scheduler
# -------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# -------------------------
# Training and Validation
# -------------------------
best_val_acc = 0.0

def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"[Train] Epoch {epoch+1}/{NUM_EPOCHS}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")
    return epoch_loss, epoch_acc


def validate(epoch):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    val_loss = running_loss / len(val_ds)
    val_acc = (all_preds == all_labels).mean()
    print(f"[Val]   Epoch {epoch+1}/{NUM_EPOCHS}  loss={val_loss:.4f}  acc={val_acc:.4f}")
    return val_loss, val_acc, all_labels, all_preds


# -------------------------
# 主训练循环
# -------------------------
for epoch in range(NUM_EPOCHS):
    train_one_epoch(epoch)
    val_loss, val_acc, y_true, y_pred = validate(epoch)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "le_classes": le.classes_
        }, SAVE_BEST_MODEL)
        print(f"Saved best model (acc={best_val_acc:.4f}) -> {SAVE_BEST_MODEL}")

    scheduler.step()

# -------------------------
# 最终验证 + 报告
# -------------------------
print("\n===== Best model evaluation and classification report =====")
checkpoint = torch.load(SAVE_BEST_MODEL, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
for name, param in model.named_parameters():
    if "blocks" in name and int(name.split('.')[1]) < 8:  # 冻结前8层
        param.requires_grad = False


model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds, all_labels = np.array(all_preds), np.array(all_labels)

print("Accuracy:", accuracy_score(all_labels, all_preds))
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=[str(c) for c in le.classes_], digits=4))

print(f"\nTraining finished. Best val acc = {best_val_acc:.4f}")
