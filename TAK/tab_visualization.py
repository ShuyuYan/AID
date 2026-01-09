import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_excel("/home/yanshuyu/Data/AID/all.xlsx", sheet_name='714')
features = [
    "esr", "crp", "saa",
    "il1beta", "il6", "il8", "il2r", "il10", "tnf",
    "igg", "igm", "iga", "ige",
    "c3", "c4", "ch50"
]

# Z-score标准化
df_scaled = df.copy()
for col in features:
    df_scaled[col] = (df[col] - df[col].mean()) / df[col].std()

# 剔除大于10的离群点
for col in features:
    df_scaled.loc[df_scaled[col] > 9, col] = float('nan')

plot_data = df_scaled.melt(id_vars=['pred'], value_vars=features,
                           var_name='Features', value_name='Z-Score Value').dropna()

sns.set_theme(style="white")
plt.figure(figsize=(24, 10))  # 稍微增加宽度以容纳更多标注

ax = sns.boxplot(x='Features', y='Z-Score Value', hue='pred', data=plot_data,
                 hue_order=[0, 1, 2],
                 palette=['red', 'yellow', 'blue'],
                 width=0.7,
                 linewidth=1.5,
                 fliersize=2)


# 辅助函数：将p值转换为星号
def get_p_text(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return None


# 遍历每个指标进行两两比较
for i, feature in enumerate(features):
    # 提取三组原始数据（用于统计）
    g0 = df[df['pred'] == 0][feature].dropna()
    g1 = df[df['pred'] == 1][feature].dropna()
    g2 = df[df['pred'] == 2][feature].dropna()

    # 提取标准化后的数据最大值（用于确定标注高度）
    curr_max = df_scaled[feature].max()

    # 定义比较对和横线高度偏移
    # x坐标修正：seaborn boxplot 中，同一组内的三个箱体中心分别在 i-0.26, i, i+0.26 附近
    pairs = [
        (0, 1, i - 0.26, i, curr_max + 0.5),  # 0 vs 1
        (1, 2, i, i + 0.26, curr_max + 1),  # 1 vs 2
        (0, 2, i - 0.26, i + 0.26, curr_max + 1.5)  # 0 vs 2
    ]

    groups = {0: g0, 1: g1, 2: g2}

    for p_idx1, p_idx2, x1, x2, y_pos in pairs:
        # 执行 Mann-Whitney U 检验
        if len(groups[p_idx1]) > 0 and len(groups[p_idx2]) > 0:
            _, p_val = stats.mannwhitneyu(groups[p_idx1], groups[p_idx2])
            p_text = get_p_text(p_val)

            if p_text:
                # 绘制横线
                plt.plot([x1, x1, x2, x2], [y_pos, y_pos + 0.1, y_pos + 0.1, y_pos], lw=0.7, c='black')
                # 绘制星号
                plt.text((x1 + x2) / 2, y_pos + 0.1, p_text, ha='center', va='bottom', fontsize=10)

for i in range(len(features)):
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5, color="#E6E6E6", alpha=0.25, zorder=0)

plt.title('Pairwise Comparison of 16 Indicators (Mann-Whitney U Test)', fontsize=16)
plt.ylabel('Standardized Value (Z-Score)', fontsize=12)
plt.xlabel('Indicators', fontsize=12)
plt.legend(title='Prediction Class', bbox_to_anchor=(1.01, 1), loc='upper left')
ax.set_xlim(-0.5, len(features) - 0.5)

# 自动调整y轴范围，防止标注出界
plt.ylim(plot_data['Z-Score Value'].min() - 0.3, plot_data['Z-Score Value'].max() + 2)
plt.tight_layout()
plt.show()
