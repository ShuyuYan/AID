import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torchvision.models as models


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

