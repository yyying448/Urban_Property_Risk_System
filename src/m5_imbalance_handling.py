import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. 构造极端不平衡数据 (950个好人, 50个坏人)
np.random.seed(42)
X_good = np.random.normal(2, 1, (950, 2))
X_bad = np.random.normal(4, 1, (50, 2))
X = np.vstack([X_good, X_bad])
y = np.array([0]*950 + [1]*50)

# 2. 方案 A：直接训练 (模型会忽略坏人)
model_normal = LogisticRegression()
model_normal.fit(X, y)
y_pred_normal = model_normal.predict(X)

# 3. 方案 B：手写阈值移动 (公式 3.48)
probs = model_normal.predict_proba(X)[:, 1]
# 银行业务决策：设定代价比例，假设漏杀代价是误杀的 5 倍
cost_ratio = 5 
custom_threshold = 1 / (1 + cost_ratio) 
y_pred_threshold = (probs > custom_threshold).astype(int)

# 4. 结果对比
print("--- 方案 A: 直接预测 (默认阈值 0.5) ---")
print(f"抓出的坏人数量: {sum(y_pred_normal & y == 1)} / 50")
print(classification_report(y, y_pred_normal, target_names=['好人', '坏人']))

print("\n--- 方案 B: 阈值移动 (代价敏感阈值) ---")
print(f"抓出的坏人数量: {sum(y_pred_threshold & y == 1)} / 50")
print(classification_report(y, y_pred_threshold, target_names=['好人', '坏人']))