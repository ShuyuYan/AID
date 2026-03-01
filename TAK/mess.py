import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms

# 1. 读取数据
file_path = "/home/yanshuyu/Data/AID/all.xlsx"
df = pd.read_excel(file_path, sheet_name='in')

features = [
    "ESR", "CRP", "SAA", "IL-6", "IL-8", "IL-2R", "IL-10", "TNF-α",
    "IgG", "IgM", "IgA", "IgE", "C3", "C4", "CH50", "WBC", "Neutrophil",
    "Lymphocyte", "Monocyte", "CD19", "CD4", "CD8", "CD4/CD8"
]

# 2. 数据处理
valid_features = [f for f in features if f in df.columns]
if len(valid_features) < len(features):
    print("注意：部分列名在Excel中未��到。")

# 按 type 分组求均值 (结果：行=0/1/2, 列=Features)
df_grouped = df.groupby('type')[valid_features].mean()

# 映射行索引名字
mapping = {0: 'Treatment A', 1: 'Treatment B', 2: 'Treatment C'}
df_grouped.index = df_grouped.index.map(lambda x: mapping.get(x, x))

# 3. Z-Score 标准化 (重要修改：axis=0)
# 我们需要对比的是"同一个指标"在不同Treatment间的高低
# 因为现在指标是"列"，所以要按列(axis=0)进行标准化
plot_data_norm = df_grouped.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

# 4. 绘图
# 自定义颜色: 蓝-白-紫
custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_style",
                                                        ["#00BFFF", "#FFFFFF", "#9370DB"])

# 调整画布：宽长型
plt.figure(figsize=(16, 6))

ax = sns.heatmap(plot_data_norm,
                 cmap=custom_cmap,
                 annot=False,
                 linewidths=1,
                 linecolor='white',
                 center=0,
                 cbar_kws={'label': 'Z-Score', 'shrink': 0.8, 'pad': 0.02},
                 square=False)

# 5. 调整坐标轴
plt.xlabel('Indicators', fontsize=20)
plt.ylabel('')
plt.yticks(rotation=0, fontsize=14)
plt.xticks(rotation=45, fontsize=16)

plt.tight_layout()
plt.savefig('heatmap_horizontal.tiff', dpi=300, bbox_inches='tight')
plt.show()
