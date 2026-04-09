import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 构造四分类模拟数据 (A, B, C, D)
np.random.seed(42)
def generate_level_data(center, label, size=100):
    data = np.random.multivariate_normal(center, [[0.3, 0], [0, 0.3]], size)
    return data, np.full(size, label)

# 模拟不同风险等级的分布
X_A, y_A = generate_level_data([2, 2], 0) # A级：双高
X_B, y_B = generate_level_data([4, 2], 1) # B级
X_C, y_C = generate_level_data([2, 4], 2) # C级
X_D, y_D = generate_level_data([5, 5], 3) # D级：双低

X = np.vstack([X_A, X_B, X_C, X_D])
y = np.concatenate([y_A, y_B, y_C, y_D])

# 2. 手写 OvO 投票机制 (对应项目书 3.4.2) 
classes = [0, 1, 2, 3]
n_classes = len(classes)
votes = np.zeros((X.shape[0], n_classes))

print("--- 开始 OvO 训练 (共需 6 个二分类器) ---")
for i in range(n_classes):
    for j in range(i + 1, n_classes):
        # 挑选出属于类 i 和类 j 的样本
        mask = np.isin(y, [i, j])
        X_sub, y_sub = X[mask], y[mask]
        
        # 训练二分类器
        clf = LogisticRegression()
        clf.fit(X_sub, y_sub)
        
        # 对全体样本进行预测，胜者得一票
        pred = clf.predict(X)
        for idx, p in enumerate(pred):
            votes[idx, p] += 1
        print(f"分类器 {i} vs {j} 训练完成")

# 最终预测：得票最多的类别
y_pred = np.argmax(votes, axis=1)

# 3. 绘制四分类混淆矩阵 (项目书要求产出) [cite: 104]
plt.figure(figsize=(8, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['A', 'B', 'C', 'D'], yticklabels=['A', 'B', 'C', 'D'])
plt.title("M4: 风险等级评定混淆矩阵 (OvO 策略)")
plt.xlabel("预测等级"); plt.ylabel("实际等级")
plt.show()