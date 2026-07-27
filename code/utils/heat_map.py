import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
# 你的 10x10 参数矩阵
performance_matrix = np.array([
    [42.213, 36.977, 31.700, 29.475, 28.695, 27.469, 27.129, 26.830, 26.596, 25.463],
    [42.180, 37.322, 32.234, 29.855, 29.027, 27.713, 27.338, 27.014, 26.759, 25.559],
    [42.054, 35.291, 34.952, 31.844, 30.539, 28.678, 28.144, 27.098, 27.366, 25.940],
    [41.360, 32.600, 33.925, 33.281, 32.142, 29.907, 29.189, 28.593, 28.160, 26.442],
    [40.938, 31.740, 32.721, 33.031, 32.584, 30.722, 30.005, 29.395, 28.893, 26.934],
    [40.613, 30.996, 31.259, 31.898, 32.128, 31.128, 30.610, 30.120, 29.630, 27.447],
    [40.703, 30.876, 30.680, 31.213, 31.559, 31.032, 30.677, 30.310, 29.888, 27.748],
    [40.543, 30.604, 30.221, 30.720, 31.123, 30.854, 30.614, 30.340, 29.973, 28.021],
    [40.570, 30.548, 30.041, 30.492, 30.913, 30.751, 30.559, 30.327, 29.991, 28.155],
    [40.761, 30.532, 29.462, 29.754, 30.159, 30.105, 30.174, 30.132, 29.934, 28.372],
])

col_max = np.max(performance_matrix, axis=0)  # shape (10,)

# 每列除以对应最大值（广播机制）
normalized_matrix = performance_matrix / col_max
# 直接使用绘图代码
sns.set(font_scale=1.2)
plt.figure(figsize=(8,6))
# white_orange = LinearSegmentedColormap.from_list('white_orange', ['orange', 'white'])

ax = sns.heatmap(
    normalized_matrix,
    # annot=True,
    # fmt=".2f",
    cmap="Oranges_r",
    cbar=True,
    square=True
)

ax.set_xlabel("Prompt Index")
ax.set_ylabel("Image Index")
ax.set_title("Performance Heatmap (Diagonal indicates matched pairs)")

# 高亮对角线
# size = performance_matrix.shape[0]
# for i in range(size):
#     ax.scatter(i+0.5, i+0.5, s=100, c='red', marker='o', edgecolors='black')

plt.tight_layout()
plt.show()