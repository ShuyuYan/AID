import pandas as pd
import os
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import xgboost as xgb

# 读取数据
df = pd.read_excel(os.path.expanduser('~/Data/AID/all.xlsx'), sheet_name='pre')

# 标签列
target_col = df.columns[-1]

# 只保留数值特征
X = df.select_dtypes(include=["int64", "float64"])
X = X.drop(columns=[target_col], errors="ignore")
y = df[target_col]

# 删除全NaN列
X = X.dropna(axis=1, how="all")

# 缺失值填充（中位数）
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# 转换为numpy
X_np = X_imputed
y_np = y.values

# =================== 先做一次数据划分（保持一致对比） ===================
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42, stratify=y
)

# =============== 特征选择 (Boruta) ===============
rf_boruta = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
boruta_selector = BorutaPy(
    estimator=rf_boruta,
    n_estimators='auto',
    random_state=42
)

print("正在运行Boruta特征选择...")
boruta_selector.fit(X_np, y_np)

# 宽松选择：Confirmed + Weak
selected_mask = boruta_selector.support_ | boruta_selector.support_weak_
selected_features = X.columns[selected_mask].tolist()
print("选中的特征数量:", len(selected_features))

# 提取选择后的特征
X_selected = X_np[:, selected_mask]

# 划分训练测试集（保持 stratify 一致）
X_train_sel, X_test_sel, _, _ = train_test_split(
    X_selected, y_np, test_size=0.2, random_state=42, stratify=y
)

# =================== 四个模型对比 ===================

# 1. RF (原始特征)
clf_rf_full = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
clf_rf_full.fit(X_train_full, y_train)
y_pred_rf_full = clf_rf_full.predict(X_test_full)
print("\n=== 随机森林 (原始特征) ===")
print("准确率:", accuracy_score(y_test, y_pred_rf_full))
print("分类报告:\n", classification_report(y_test, y_pred_rf_full))

# 2. RF (Boruta特征)
clf_rf_sel = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
clf_rf_sel.fit(X_train_sel, y_train)
y_pred_rf_sel = clf_rf_sel.predict(X_test_sel)
print("\n=== 随机森林 (Boruta特征) ===")
print("准确率:", accuracy_score(y_test, y_pred_rf_sel))
print("分类报告:\n", classification_report(y_test, y_pred_rf_sel))

# 3. XGBoost (原始特征)
clf_xgb_full = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=1,
    eval_metric="mlogloss",
    use_label_encoder=False
)
clf_xgb_full.fit(X_train_full, y_train)
y_pred_xgb_full = clf_xgb_full.predict(X_test_full)
print("\n=== XGBoost (原始特征) ===")
print("准确率:", accuracy_score(y_test, y_pred_xgb_full))
print("分类报告:\n", classification_report(y_test, y_pred_xgb_full))

# 4. XGBoost (Boruta特征)
clf_xgb_sel = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=1,
    eval_metric="mlogloss",
    use_label_encoder=False
)
clf_xgb_sel.fit(X_train_sel, y_train)
y_pred_xgb_sel = clf_xgb_sel.predict(X_test_sel)
print("\n=== XGBoost (Boruta特征) ===")
print("准确率:", accuracy_score(y_test, y_pred_xgb_sel))
print("分类报告:\n", classification_report(y_test, y_pred_xgb_sel))
