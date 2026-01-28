import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_excel("/home/yanshuyu/Data/AID/all.xlsx", sheet_name='in')
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

plot_data = df_scaled.melt(id_vars=['type'], value_vars=features,
                           var_name='Features', value_name='Z-Score Value').dropna()

sns.set_theme(style="white")
plt.figure(figsize=(24, 10))  # 稍微增加宽度以容纳更多标注

ax = sns.boxplot(x='Features', y='Z-Score Value', hue='type', data=plot_data,
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
    g0 = df[df['type'] == 0][feature].dropna()
    g1 = df[df['type'] == 1][feature].dropna()
    g2 = df[df['type'] == 2][feature].dropna()

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

    kw_significant = False
    if len(g0) > 0 and len(g1) > 0 and len(g2) > 0:
        try:
            _, p_kw = stats.kruskal(g0, g1, g2)
            if p_kw < 0.05:
                kw_significant = True
        except ValueError:
            pass  # 数据异常时跳过

    # 步骤2：只有当全局检验显著时，才进行两两比较 (Post-hoc)
    if kw_significant:
        for p_idx1, p_idx2, x1, x2, y_pos in pairs:
            if len(groups[p_idx1]) > 0 and len(groups[p_idx2]) > 0:
                _, p_val = stats.mannwhitneyu(groups[p_idx1], groups[p_idx2])

                # Bonferroni 修正：P值乘以比较次数（3次）
                p_val_corrected = p_val * 3

                if p_val_corrected > 1: p_val_corrected = 1
                p_text = get_p_text(p_val_corrected)

                if p_text:
                    # 绘制横线
                    plt.plot([x1, x1, x2, x2], [y_pos, y_pos + 0.1, y_pos + 0.1, y_pos], lw=0.7, c='black')
                    # 绘制星号
                    plt.text((x1 + x2) / 2, y_pos + 0.1, p_text, ha='center', va='bottom', fontsize=14)

for i in range(len(features)):
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5, color="#E6E6E6", alpha=0.25, zorder=0)

plt.ylabel('Standardized Value (Z-Score)', fontsize=22)
plt.xlabel('Indicators', fontsize=22)
plt.tick_params(axis='x', labelsize=22)
plt.tick_params(axis='y', labelsize=22)
legend = plt.legend(title='Prediction Treatment', loc='upper left', fontsize=16, title_fontsize=18)
ax.set_xlim(-0.5, len(features) - 0.5)
labels = ['Treatment A', 'Treatment B', 'Treatment C']
for text, label in zip(legend.get_texts(), labels):
    text.set_text(label)
plt.ylim(plot_data['Z-Score Value'].min() - 0.3, plot_data['Z-Score Value'].max() + 2)
plt.tight_layout()
plt.savefig('tab_visualization.png', dpi=1000)
plt.show()
