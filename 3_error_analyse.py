import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

"""
误差分析：
    1. 绘制混淆矩阵
"""
cm_data = np.array([[241, 13], [6, 140]])
norm_cm_data = cm_data / cm_data.sum(axis=1, keepdims=True)
np.fill_diagonal(norm_cm_data, 0)
cm_df = pd.DataFrame(cm_data, index=["Negative", "Positive"],columns=["Negative", "Positive"])
norm_cm_df = pd.DataFrame(norm_cm_data, index=["Negative", "Positive"],columns=["Negative", "Positive"])

f, ax = plt.subplots()
sns.heatmap(cm_df, annot=True, fmt="d")
# plt.matshow(norm_cm_data, cmap=plt.cm.gray)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predict') 
ax.set_ylabel('Target')
plt.savefig('confusion_matrix.png', dpi=200)