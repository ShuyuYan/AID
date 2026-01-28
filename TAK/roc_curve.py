import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
"""
使用不同数量模态的模型ROC对比图
"""

y_true = np.load("/home/yanshuyu/Data/AID/results/roc/y_true.npy")   # shape (N,)
model_files = [
    ("Multimodal", "/home/yanshuyu/Data/AID/results/roc/multimodal.npz"),
    ("Bimodal (Image + Report)", "/home/yanshuyu/Data/AID/results/roc/image_report.npz"),
    ("Bimodal (Image + SCD)", "/home/yanshuyu/Data/AID/results/roc/image_tab.npz"),
    ("Bimodal (Report + SCD)", "/home/yanshuyu/Data/AID/results/roc/report_tab2.npz"),
    ("Unimodal (Image)", "/home/yanshuyu/Data/AID/results/roc/image.npz"),
    ("Unimodal (Report)", "/home/yanshuyu/Data/AID/results/roc/report.npz"),
    ("Unimodal (SCD)", "/home/yanshuyu/Data/AID/results/roc/tab.npz"),
]
k = 0

y_true_k = (y_true == k).astype(int)

plt.figure(figsize=(7, 6))

for model_name, file in model_files:
    data = np.load(file, allow_pickle=True)
    y_score = data["y_score"]          # (N, 3)
    y_score_k = y_score[:, k]          # 只取第 k 类概率

    fpr, tpr, _ = roc_curve(y_true_k, y_score_k)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (AUC = {roc_auc:.3f})"
    )

# 对角线
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve Treatment A")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc.png', dpi=1000, bbox_inches='tight')
plt.show()
