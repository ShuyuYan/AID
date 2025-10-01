import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, make_scorer
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ========== 数据读取 ==========
df = pd.read_excel(os.path.expanduser('~/Data/AID/all.xlsx'), sheet_name='data')
target_col = df.columns[-3]

# 特征 + 标签
X = df.select_dtypes(include=["int64", "float64"])
X = X.drop(columns=[target_col], errors="ignore")
y = df[target_col].values

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ========== 定义评估指标 ==========
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "f1_macro": make_scorer(f1_score, average="macro")
}

# ========== 定义模型 ==========
def get_models(class_weight=None):
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight=class_weight
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=5,
            random_state=42, use_label_encoder=False, eval_metric="mlogloss"
        )
    }

# ========== 存储结果 ==========
results = []
reports = []

# 自定义函数：在CV中收集 y_true, y_pred
def collect_reports(model, X, y, use_class_weight=False, model_name=""):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if use_class_weight and model_name == "XGBoost":
            weights = compute_sample_weight(class_weight="balanced", y=y_tr)
            model.fit(X_tr, y_tr, sample_weight=weights)
        else:
            model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        y_true_all.extend(y_te)
        y_pred_all.extend(y_pred)

    return classification_report(y_true_all, y_pred_all, digits=3)


# ---- 1. Baseline ----
for name, model in get_models().items():
    cv_res = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)

    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    results.append({
        "Setting": "Baseline",
        "Model": name,
        "Acc_mean": np.mean(cv_res["test_accuracy"]),
        "Acc_std": np.std(cv_res["test_accuracy"]),
        "F1_mean": np.mean(cv_res["test_f1_macro"]),
        "F1_std": np.std(cv_res["test_f1_macro"]),
        "Test_Acc": test_acc,
        "Test_F1": test_f1
    })
    rep = collect_reports(model, X, y, use_class_weight=False, model_name=name)
    reports.append(("Baseline", name, rep))

# ---- 2. ClassWeight ----
for name, model in get_models(class_weight="balanced").items():
    if name == "XGBoost":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        acc_scores, f1_scores = [], []
        from sklearn.utils.class_weight import compute_sample_weight
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            weights = compute_sample_weight(class_weight="balanced", y=y_tr)
            model.fit(X_tr, y_tr, sample_weight=weights)
            y_pred = model.predict(X_te)
            acc_scores.append(accuracy_score(y_te, y_pred))
            f1_scores.append(f1_score(y_te, y_pred, average="macro"))

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred, average='macro')
        results.append({
            "Setting": "ClassWeight",
            "Model": name,
            "Acc_mean": np.mean(acc_scores),
            "Acc_std": np.std(acc_scores),
            "F1_mean": np.mean(f1_scores),
            "F1_std": np.std(f1_scores),
            "Test_Acc": test_acc,
            "Test_F1": test_f1
        })
        rep = collect_reports(model, X, y, use_class_weight=True, model_name=name)
        reports.append(("ClassWeight", name, rep))
    else:
        cv_res = cross_validate(model, X, y, cv=5, scoring=scoring)

        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='macro')
        results.append({
            "Setting": "ClassWeight",
            "Model": name,
            "Acc_mean": np.mean(cv_res["test_accuracy"]),
            "Acc_std": np.std(cv_res["test_accuracy"]),
            "F1_mean": np.mean(cv_res["test_f1_macro"]),
            "F1_std": np.std(cv_res["test_f1_macro"]),
            "Test_Acc": test_acc,
            "Test_F1": test_f1
        })
        rep = collect_reports(model, X, y, use_class_weight=True, model_name=name)
        reports.append(("ClassWeight", name, rep))

# ---- 3. SMOTE ----
smote = SMOTE(random_state=42)
n_splits = 5

for name, model in get_models().items():
    fold_acc = []
    fold_f1 = []

    for i in range(n_splits):
        X_train_inner, X_val, y_train_inner, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42 + i)
        X_res, y_res = smote.fit_resample(X_train_inner, y_train_inner)
        model.fit(X_res, y_res)
        y_val_pred = model.predict(X_val)
        fold_acc.append(accuracy_score(y_val, y_val_pred))
        fold_f1.append(f1_score(y_val, y_val_pred, average='macro'))

    X_res_full, y_res_full = smote.fit_resample(X_train, y_train)
    model.fit(X_res_full, y_res_full)
    y_test_pred = model.predict(X_test)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')

    results.append({
        "Setting": "SMOTE",
        "Model": name,
        "Acc_mean": np.mean(fold_acc),
        "F1_mean": np.mean(fold_f1),
        "Acc_std": np.std(fold_acc),
        "F1_std": np.std(fold_f1),
        "Test_Acc": test_acc,
        "Test_F1": test_f1
    })
    rep = collect_reports(model, X, y, use_class_weight=False, model_name=name)
    reports.append(("SMOTE", name, rep))

# ========== 汇总 ==========
df_results = pd.DataFrame(results)
print("=== 结果对比表 ===")
print(df_results.to_string())

print("\n=== 各模型 classification report ===")
for setting, model, rep in reports:
    print(f"\n--- {setting} | {model} ---")
    print(rep)
