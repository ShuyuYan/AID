import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
import numpy as np

def multiple_imputation(file_path, sheet_name="data", n_sets=5, output_prefix="imputed"):
    # 读入数据
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 仅对数值型数据做插补
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        raise ValueError("没有数值型列可供插补")

    for i in range(1, n_sets + 1):
        # 每次用不同随机种子，保证插补结果不同
        imputer = IterativeImputer(random_state=i, max_iter=10, sample_posterior=True)

        df_imputed = df.copy()
        df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        # 处理不合理的负值：统一截断到 0
        df_imputed[numeric_cols] = df_imputed[numeric_cols].clip(lower=0)

        # 保存结果
        output_file = f"{output_prefix}_{i}.xlsx"
        df_imputed.to_excel(output_file, sheet_name=sheet_name, index=False)
        print(f"第 {i} 套插补数据已保存到 {output_file}")

# 使用示例
if __name__ == "__main__":
    multiple_imputation("/home/yanshuyu/Data/AID/new baseline.xlsx", sheet_name="Sheet2", n_sets=5,
                        output_prefix="/home/yanshuyu/Data/AID/imputed")
