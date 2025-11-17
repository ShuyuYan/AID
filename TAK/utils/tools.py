import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from boruta import BorutaPy


def pca_model(X, n_components=0.95):
    # - n_components = 0.95  → 保留95%的方差信息（推荐）
    # - n_components = 50    → 固定保留50维（可自行调整）
    model = PCA(n_components=n_components, random_state=42)
    X_pca = model.fit_transform(X)

    print(f"✅ PCA后特征维度：{X_pca.shape}，保留累计方差：{model.explained_variance_ratio_.sum():.3f}")
    return X_pca


def boruta_model(X, y):
    rf = RandomForestClassifier(
        n_jobs=-1,
        class_weight='balanced',
        max_depth=5,
        n_estimators=200,
        random_state=42
    )
    boruta = BorutaPy(
        estimator=rf,
        n_estimators='auto',
        verbose=2,
        random_state=42
    )

    boruta.fit(X, y)
    X_boruta = X[:, boruta.support_]

    print(f"✅ Boruta筛选后特征维度：{X_boruta.shape[1]}")
    print("被选中特征比例：", boruta.support_.mean())
    print("特征重要性：", boruta_model.estimator_.feature_importances_)
    print("特征排名：", boruta_model.ranking_)

