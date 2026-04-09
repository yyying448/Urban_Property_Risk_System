import numpy as np
import matplotlib.pyplot as plt

# 1. 构造更具区分度的模拟数据（模拟真实的信贷差异）
np.random.seed(42)
# 好客户：收入高、信用分高
good_clients = np.random.multivariate_normal([2, 2], [[0.5, 0.1], [0.1, 0.5]], 100)
# 坏客户：收入低、信用分低
bad_clients = np.random.multivariate_normal([5, 5], [[0.5, 0.1], [0.1, 0.5]], 100)

# 2. 核心算法：求解投影向量 w (西瓜书公式 3.33-3.39)
u1 = np.mean(good_clients, axis=0)
u2 = np.mean(bad_clients, axis=0)

# 计算 Sw (类内散度) 
Sw = (good_clients - u1).T @ (good_clients - u1) + (bad_clients - u2).T @ (bad_clients - u2)

# 计算 w (最优投影方向) [cite: 85]
w = np.linalg.inv(Sw) @ (u1 - u2).reshape(-1, 1)

# 3. 绘制带有“投影感”的业务画像
plt.figure(figsize=(10, 7))
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 散点图
plt.scatter(good_clients[:,0], good_clients[:,1], label='好客户 (低风险)', alpha=0.6, c='#3498db')
plt.scatter(bad_clients[:,0], bad_clients[:,1], label='坏客户 (高风险)', alpha=0.6, c='#e74c3c', marker='x')

# 绘制 LDA 判别线：这条线应该能把两类人切开
# 我们画一条垂直于 w 的决策线，这才是银行审批时的“分水岭”
x_axis = np.linspace(0, 7, 100)
# 计算决策边界（中点位置）
center = (u1 + u2) / 2
decision_slope = -w[0][0] / w[1][0] # 垂直于 w 的斜率
y_axis = decision_slope * (x_axis - center[0]) + center[1]

plt.plot(x_axis, y_axis, 'k--', label='审批分水岭 (Decision Boundary)')
plt.fill_between(x_axis, y_axis, 8, color='#e74c3c', alpha=0.05, label='高风险区')
plt.fill_between(x_axis, 0, y_axis, color='#3498db', alpha=0.05, label='低风险区')

plt.xlim(0, 8); plt.ylim(0, 8)
plt.title("M3: 客户风险群体画像 (LDA 判别分析)")
plt.xlabel("特征 A (如：账户余额标准化)"); plt.ylabel("特征 B (如：信用分标准化)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()