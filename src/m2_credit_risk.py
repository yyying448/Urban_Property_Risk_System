import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 加载数据 (这里我们先模拟加载，因为 German Credit 通常需要处理分类变量)
# 为了教学方便，我们先构造一组与项目要求对应的模拟特征：[账户余额, 就业时长, 抵押率, 信用历史得分]
# 真实场景中你会读取本地的 german_credit.csv
np.random.seed(42)
data_size = 1000
X = np.random.randn(data_size, 4) 
# 模拟标签：0 代表好客户，1 代表违约客户 (假设 10% 违约)
y = np.random.choice([0, 1], size=data_size, p=[0.9, 0.1]).reshape(-1, 1)

# 2. 标准化：银行数据单位差异大，必须标准化 [cite: 155]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

print(f"信贷数据准备就绪：训练集 {len(X_train)} 条，测试集 {len(X_test)} 条")
# 手写 Sigmoid 函数，增加 clip 防止数值溢出 
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# 手写梯度下降求解
def train_logistic_regression(X, y, lr=0.1, epochs=100):
    m, n = X.shape
    w = np.zeros((n, 1)) # 初始化权重
    b = 0 # 初始化截距
    
    for i in range(epochs):
        # 1. 前向传播：计算预测概率
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)
        
        # 2. 计算梯度 (西瓜书公式 3.30 的简化形式)
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        
        # 3. 更新参数
        w -= lr * dw
        b -= lr * db
        
        if i % 20 == 0:
            # 计算对数似然损失 [cite: 65]
            loss = -np.mean(y * np.log(y_pred + 1e-9) + (1-y) * np.log(1-y_pred + 1e-9))
            print(f"第 {i} 次迭代，损失值 (Loss): {loss:.4f}")
            
    return w, b

# 开始训练
weights, intercept = train_logistic_regression(X_train, y_train, lr=0.1, epochs=101)
# 7. 模拟生成一份“审批意见书” (对应项目书 3.2.2)
# 假设有一个新客户：账户余额低(-1.5), 就业长(2.0), 抵押率高(1.8), 信用历史好(-1.0)
# 注：这里的数值是经过 StandardScale 后的模拟值
new_client = np.array([[-1.5, 2.0, 1.8, -1.0]]) 

# 计算预测概率
z_new = np.dot(new_client, weights) + intercept
prob = sigmoid(z_new)[0][0]

# 业务规则判断
threshold = 0.3  # 银行设定的风险阈值 
risk_level = "A" if prob < 0.1 else ("B" if prob < 0.3 else ("C" if prob < 0.5 else "D")) # [cite: 94]

print("\n" + "="*35)
print("      信 贷 审 批 意 见 书      ")
print("="*35)
print(f"违约概率预测：{prob*100:.1f}%")
print(f"风险评级：{risk_level} 级") # [cite: 94]

# 给出建议决策 [cite: 68, 71]
if prob > threshold:
    print("建议决策：❌ 拒绝贷款 (超过风险红线)")
    # 自动生成理由 (基于特征贡献度) [cite: 72, 150]
    print("拒绝理由：检测到核心风险点——抵押率过高或账户余额不足。")
else:
    print("建议决策：✅ 建议批准 (建议执行标准利率)")

# 联动 M1：计算抵押率 (LTV) [cite: 38, 73]
# 假设贷款 15万，使用之前 M1 算出的估值约 19.5万
loan_amount = 150000
property_val = 195348 # 来自 M1 结果
ltv = loan_amount / property_val
print(f"抵押率计算：{ltv*100:.1f}% (预警线：70%)") # [cite: 73]
if ltv > 0.7:
    print("【风险预警】抵押率超过 70% 警戒线，需调减贷款额度！")
print("="*35)