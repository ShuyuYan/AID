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


# ----------------------------
# Model definitions
# ----------------------------
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
        x = self.encoder(x)       # [B, C, 1, 1]
        x = x.flatten(1)          # [B, C]
        x = self.fc(x)            # [B, out_dim]
        return x


class TabularEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class TextEncoder(nn.Module):
    def __init__(self, bert_path, out_dim=256, freeze_bert=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_path)
        self.proj = nn.Linear(self.bert.config.hidden_size, out_dim)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.proj(pooled)


class GateFusion(nn.Module):
    def __init__(self, dim_a=256, dim_b=256, hidden_dim=256):
        super().__init__()
        self.gate_fc = nn.Sequential(
            nn.Linear(dim_a + dim_b, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_a),
            nn.Sigmoid()
        )

    def forward(self, a, b):
        # a,b: [B, D] (same D)
        x = torch.cat([a, b], dim=1)
        gate = self.gate_fc(x)
        fused = gate * a + (1 - gate) * b
        return fused


class ImageTabTextModel(nn.Module):
    def __init__(self, num_labels, tab_dim, bert_path,
                 img_backbone="resnet18", feature_dim=256, tab_out_dim=128):
        super().__init__()
        self.image_encoder = ImageEncoder(backbone=img_backbone, out_dim=feature_dim)
        self.tab_encoder = TabularEncoder(in_dim=tab_dim, out_dim=tab_out_dim)
        # project tab to feature_dim
        self.tab_proj = nn.Linear(tab_out_dim, feature_dim)
        self.text_encoder = TextEncoder(bert_path, out_dim=feature_dim, freeze_bert=False)

        # two-stage fusion: image+tab -> then fuse with text
        self.fusion_im_tab = GateFusion(dim_a=feature_dim, dim_b=feature_dim, hidden_dim=feature_dim)
        self.fusion_all = GateFusion(dim_a=feature_dim, dim_b=feature_dim, hidden_dim=feature_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, num_labels)
        )

    def forward(self, image, tab, input_ids=None, attention_mask=None):
        img_feat = self.image_encoder(image)          # [B, D]
        tab_feat = self.tab_encoder(tab)              # [B, tab_out_dim]
        tab_feat = self.tab_proj(tab_feat)            # [B, D]
        im_tab = self.fusion_im_tab(img_feat, tab_feat)  # [B, D]

        if input_ids is None:
            # if no text provided, use zeros
            text_feat = torch.zeros_like(im_tab).to(im_tab.device)
        else:
            text_feat = self.text_encoder(input_ids, attention_mask)  # [B, D]

        fused = self.fusion_all(im_tab, text_feat)    # [B, D]
        logits = self.classifier(fused)               # [B, C]
        return logits, fused


# ----------------------------
# Helper functions
# ----------------------------
def sharpen(probs, T=0.5):
    probs = probs.clamp(min=1e-7)
    p_power = probs ** (1.0 / T)
    p_sharp = p_power / p_power.sum(dim=1, keepdim=True)
    return p_sharp


@torch.no_grad()
def update_ema(student, teacher, alpha):
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(alpha).add_(s_param.data * (1 - alpha))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_path = "/home/yanshuyu/Data/AID/TAK/Bio_ClinicalBERT"
    excel_path = "/home/yanshuyu/Data/AID/all.xlsx"
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    writer = SummaryWriter(log_dir="/home/yanshuyu/Data/AID/runs/Multimodal")
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "/home/yanshuyu/Data/AID/TAK/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_excel(excel_path, sheet_name='newlable')
    label_col = df.columns[-1]

    max_length = 384

    X = df.select_dtypes(include=['int64', 'float64'])
    X = X.drop(columns=[label_col], errors='ignore')
    imputer = SimpleImputer(strategy="mean")
    X_np = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_np)
    report = df['mra_examination_re_des_1'].astype(str).tolist()[:len(X_np)]
    labels_series = df[label_col].astype(int)
    num_labels = 3
    valid_label_values = [0, 1, 2]

    y_for_dataset = labels_series.values  # contains -1 for unlabeled
    data = TADataset(df, report, X_np, y_for_dataset, tokenizer, max_length)

    # ---------- create train/val indices: ensure val contains ONLY labeled samples ----------
    all_indices = df.index.to_numpy()

    labeled_indices = df.index[df[label_col] != -1].to_numpy()
    unlabeled_indices = df.index[df[label_col] == -1].to_numpy()

    # stratified split on labeled indices
    train_lab_idx, val_lab_idx = train_test_split(
        labeled_indices,
        test_size=0.2,
        random_state=42,
        stratify=df.loc[labeled_indices, label_col].values
    )

    # final train indices = train_lab_idx + unlabeled_indices
    train_indices = list(train_lab_idx) + list(unlabeled_indices.tolist())
    val_indices = list(val_lab_idx)

    # Build Subset dataloaders
    batch_size = 8
    num_workers = 4
    train_subset = Subset(data, train_indices)
    val_subset = Subset(data, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # ---------- model / teacher-student / optimizer ----------
    tab_dim = X_np.shape[1]
    student = ImageTabTextModel(
        num_labels=num_labels,
        tab_dim=tab_dim,
        bert_path=bert_path,
        img_backbone="resnet18",
        feature_dim=256,
        tab_out_dim=128
    ).to(device)

    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    lr = 2e-4
    num_epochs = 40
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-5)
    # class_weights = torch.tensor([1.0, 2.0, 4.0]).to(device)
    criterion = nn.CrossEntropyLoss()
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # semi-supervised hyperparams
    ema_alpha = 0.995
    T_sharpen = 0.5
    high_th = 0.9
    mid_th = 0.6
    weight_soft = 0.3
    weight_consistency = 0.5
    weight_pseudo_hard = 1.0
    global_step = 0
    best_val_acc = 0.85

    # ---------- training loop ----------
    for epoch in range(1, num_epochs + 1):  # num_epochs = 50
        student.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch} [Train]'):
            head = batch['head'].to(device)                # image tensor
            tab = batch['tab'].to(device).float()         # tabular features
            label = batch['label'].to(device)             # may contain -1

            # text tokens
            text_tokens = batch.get('text_tokens', None)
            if text_tokens is not None and text_tokens.get('input_ids', None) is not None:
                input_ids = text_tokens['input_ids'].to(device)
                attention_mask = text_tokens['attention_mask'].to(device)
            else:
                input_ids = None
                attention_mask = None

            optimizer.zero_grad()

            # ---- student forward ----
            student_logits, _ = student(head, tab, input_ids, attention_mask)   # [B, C]
            student_probs = torch.softmax(student_logits, dim=1)

            # ---- teacher forward ----
            with torch.no_grad():
                teacher_logits, _ = teacher(head, tab, input_ids, attention_mask)
                teacher_probs = torch.softmax(teacher_logits, dim=1)

            # ---- supervised loss on labeled ----
            mask_valid = (label != -1)
            if mask_valid.sum() > 0:
                loss_sup = criterion(student_logits[mask_valid], label[mask_valid])
            else:
                loss_sup = torch.tensor(0.0, device=device)

            # ---- pseudo & consistency for unlabeled ----
            mask_invalid = (label == -1)
            loss_pseudo = torch.tensor(0.0, device=device)
            loss_consistency = torch.tensor(0.0, device=device)

            if mask_invalid.sum() > 0:
                t_probs = teacher_probs[mask_invalid]   # [K, C]
                s_logits_unl = student_logits[mask_invalid]
                s_probs_unl = student_probs[mask_invalid]

                # sharpen teacher probs
                t_sharp = sharpen(t_probs, T=T_sharpen)

                # high confidence (teacher) -> hard pseudo
                t_max, t_pred = torch.max(t_probs, dim=1)
                mask_high = (t_max >= high_th)
                if mask_high.sum() > 0:
                    loss_pseudo = loss_pseudo + weight_pseudo_hard * criterion(
                        s_logits_unl[mask_high],
                        t_pred[mask_high]
                    )

                # medium confidence -> soft target KL
                mask_mid = (t_max < high_th) & (t_max >= mid_th)
                if mask_mid.sum() > 0:
                    s_probs_mid = s_probs_unl[mask_mid]
                    t_soft_mid = t_sharp[mask_mid].detach()
                    loss_kl_mid = kl_loss_fn(torch.log(s_probs_mid + 1e-12), t_soft_mid)
                    loss_pseudo = loss_pseudo + weight_soft * loss_kl_mid

                # overall consistency across unlabeled (student vs teacher_sharp)
                loss_consistency = weight_consistency * kl_loss_fn(torch.log(s_probs_unl + 1e-12), t_sharp.detach())

            # ---- total loss ----
            loss = loss_sup + loss_pseudo + loss_consistency
            loss.backward()
            optimizer.step()

            # EMA update of teacher
            update_ema(student, teacher, ema_alpha)

            total_loss += loss.item()
            global_step += 1


        # ---------- validation (only labeled samples) ----------
        student.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch} [Val]'):
                head = batch['head'].to(device)
                tab = batch['tab'].to(device).float()
                label = batch['label'].to(device)

                text_tokens = batch.get('text_tokens', None)
                if text_tokens is not None and text_tokens.get('input_ids', None) is not None:
                    input_ids = text_tokens['input_ids'].to(device)
                    attention_mask = text_tokens['attention_mask'].to(device)
                else:
                    input_ids = None
                    attention_mask = None

                mask_valid = (label != -1)
                if mask_valid.sum() == 0:
                    continue

                logits, _ = student(head, tab, input_ids, attention_mask)
                val_loss += criterion(logits[mask_valid], label[mask_valid]).item()

                preds = torch.argmax(logits[mask_valid], dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(label[mask_valid].cpu().tolist())

                correct += (preds == label[mask_valid]).sum().item()
                total += mask_valid.sum().item()

        if total > 0:
            val_acc = correct / total
            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
        else:
            val_acc = 0.0
            avg_train_loss = total_loss / max(1, len(train_loader))
            avg_val_loss = 0.0

        writer.add_scalar('Train Loss', avg_train_loss, epoch)
        writer.add_scalar('Val Loss', avg_val_loss, epoch)
        writer.add_scalar('Val Acc', val_acc, epoch)

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} |"
              f" Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print("\n=========== Classification Report ===========")
            print(classification_report(all_labels, all_preds, labels=[0, 1, 2],
                                        target_names=[str(c) for c in [0, 1, 2]]))

            ckpt_name = f"{start_time}_epoch{epoch}_acc{val_acc:.4f}.pth"
            best_path = os.path.join(save_dir, ckpt_name)
            torch.save(student.state_dict(), best_path)

        scheduler.step(avg_val_loss)

    writer.close()
    print("Training finished. Best val acc:", best_val_acc)
