import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def pca(X, n_components=0.95):
    # - n_components = 0.95  → 保留95%的方差信息（推荐）
    # - n_components = 50    → 固定保留50维（可自行调整）
    model = PCA(n_components=n_components, random_state=42)
    X_pca = model.fit_transform(X)

    print(f"✅ PCA后特征维度：{X_pca.shape}，保留累计方差：{model.explained_variance_ratio_.sum():.3f}")
    return X_pca
