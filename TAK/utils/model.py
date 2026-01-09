import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torchvision.models as models
from tabpfn import TabPFNClassifier


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
        x = self.encoder(x)  # [B, C, 1, 1]
        x = x.flatten(1)  # [B, C]
        x = self.fc(x)  # [B, out_dim]
        return x


class TabMLPEncoder(nn.Module):
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


class TabPFNEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=128, n_ensemble=4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.tabpfn = TabPFNClassifier(
            model_path='/home/yanshuyu/Downloads/tabpfn-v2-classifier-v2_default.ckpt',
            device='cuda' if torch.cuda. is_available() else 'cpu',
            n_estimators=n_ensemble
        )

        self.fitted = True
        self.projection = nn.Linear(512, out_dim)
        self.fallback_mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn. Dropout(0.4),
            nn.Linear(128, out_dim),
        )
        # 用于处理TabPFN特征的MLP
        self.tabpfn_mlp = None

    def fit(self, X_train, y_train):
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train. detach().cpu().numpy()

        self.tabpfn.fit(X_train, y_train)
        self.fitted = True
        return self

    def extract_tabpfn_features(self, x):
        if isinstance(x, torch.Tensor):
            device = x.device
            x_np = x. detach().cpu().numpy()
        else:
            device = 'cpu'
            x_np = x

        proba = self.tabpfn.predict_proba(x_np)
        features = torch.tensor(proba, dtype=torch.float32, device=device)
        return features

    def forward(self, x):
        if not self.fitted:
            # print('MLP')
            return self.fallback_mlp(x)

        # print('TabPFN')
        with torch.no_grad():
            tabpfn_feat = self.extract_tabpfn_features(x)
        if tabpfn_feat.shape[-1] != 512:
            self.projection = nn.Linear(
                tabpfn_feat.shape[-1], self.out_dim
            ).to(x.device)

        projected_feat = self.projection(tabpfn_feat)
        mlp_feat = self.fallback_mlp(x)

        output = projected_feat + mlp_feat
        return output


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
        # a,b:  [B, D] (same D)
        x = torch.cat([a, b], dim=1)
        gate = self.gate_fc(x)
        fused = gate * a + (1 - gate) * b
        return fused


class ImageTabTextModel(nn.Module):
    def __init__(self, num_labels, tab_dim, bert_path, img_backbone="resnet18", feature_dim=256, tab_out_dim=128,
                 n_ensemble=4):
        super().__init__()

        self.image_encoder = ImageEncoder(backbone=img_backbone, out_dim=feature_dim)

        # --- 修改:  使用 TabPFNEncoder 替换 TabularEncoder ---
        # 直接输出 feature_dim，省去额外的 projection 层
        self.tab_encoder = TabPFNEncoder(in_dim=tab_dim, out_dim=feature_dim, n_ensemble=n_ensemble)
        # 不再需要 tab_proj，因为 TabPFNEncoder 直接输出 feature_dim
        # --- 修改结束 ---

        self.text_encoder = TextEncoder(bert_path, out_dim=feature_dim, freeze_bert=False)

        self.missing_head = nn.Parameter(torch.randn(1, feature_dim))
        self.missing_thorax = nn.Parameter(torch.randn(1, feature_dim))

        # --- 修改开始:  添加 MultiheadAttention ---
        # 假设我们将融合后的图像特征、文本特征、表格特征作为序列输入
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        # --- 修改结束 ---

        self.fusion_img_img = GateFusion(dim_a=feature_dim, dim_b=feature_dim, hidden_dim=feature_dim)
        # self.fusion_img_text = GateFusion(dim_a=feature_dim, dim_b=feature_dim, hidden_dim=feature_dim) # 可以保留或替换
        # self.fusion_all = GateFusion(dim_a=feature_dim, dim_b=feature_dim, hidden_dim=feature_dim)       # 可以保留或替换

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, num_labels)
        )

    def fit_tab_encoder(self, X_train, y_train):
        """拟合 TabPFN 编码器（需要在训练前调用）"""
        self.tab_encoder.fit(X_train, y_train)
        return self

    def forward(self, head, thorax, tab, input_ids=None, attention_mask=None,
                head_mask=None, thorax_mask=None):

        head_feat = self.image_encoder(head)  # [B, D]
        thorax_feat = self.image_encoder(thorax)  # [B, D]
        if head_mask is not None:
            head_feat = head_feat * head_mask + self.missing_head * (1 - head_mask)
        if thorax_mask is not None:
            thorax_feat = thorax_feat * thorax_mask + self.missing_thorax * (1 - thorax_mask)

        # --- 修改: 直接使用 tab_encoder，不需要 tab_proj ---
        tab_feat = self.tab_encoder(tab)  # [B, feature_dim] 直接输出目标维度
        # --- 修改结束 ---

        img_fused = self.fusion_img_img(head_feat, thorax_feat)  # [B, D]

        if input_ids is None:
            text_feat = torch.zeros_like(img_fused).to(img_fused.device)
        else:
            text_feat = self.text_encoder(input_ids, attention_mask)  # [B, D]

        # 构造序列: [img_fused, text_feat, tab_feat] -> shape [B, 3, D]
        # 注意: 也可以把 head_feat 和 thorax_feat 分开不融合直接放进来，变成序列长度为4
        stack_feat = torch.stack([img_fused, text_feat, tab_feat], dim=1)  # [B, 3, D]
        attn_out, _ = self.attn(stack_feat, stack_feat, stack_feat)  # [B, 3, D]
        # 残差连接 + LayerNorm
        fused_seq = self.norm(stack_feat + attn_out)
        # 聚合策略：取平均 (Global Average Pooling) 或者取第一个token，或者Flatten
        fused = torch.mean(fused_seq, dim=1)  # [B, D]

        logits = self.classifier(fused)  # [B, C]
        return logits, fused