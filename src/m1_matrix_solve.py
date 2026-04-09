import pandas as pd
import numpy as np

# 1. 读取本地数据
df = pd.read_csv(r'D:\Urban_Property_Risk_System\data\ames_housing.csv')

# 2. 准备多维特征 (业务选型：面积、质量、房龄、地下室面积、车库面积)
features = ['GrLivArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GarageArea']
X = df[features].fillna(0).values # 转换为矩阵并处理缺失值 [cite: 49]
y = df['SalePrice'].values.reshape(-1, 1) # 转换为列向量

# 3. 构造增广矩阵 (给 X 加上一列全为 1 的列，用来算截距 b)
X_hat = np.column_stack((X, np.ones(X.shape[0])))

# 4. 手写 Ridge 正则化闭式解: w* = (X.T @ X + lambda * I)^-1 @ X.T @ y 
lam = 0.1  # 惩罚系数 lambda
I = np.eye(X_hat.shape[1]) # 单位矩阵
# 公式落地
w_star = np.linalg.inv(X_hat.T @ X_hat + lam * I) @ X_hat.T @ y

print("--- 手写矩阵模型参数 (w*) ---")
for feat, weight in zip(features + ['Intercept'], w_star.flatten()):
    print(f"{feat}: {weight:.2f}")

# 5. 模拟一个新房子的估值 [cite: 21]
# 假设：面积 1500, 质量 7, 房龄 20, 地下室 800, 车库 400
new_house = np.array([[1500, 7, 2000, 800, 400, 1]]) # 最后一位 1 对应截距
prediction = new_house @ w_star
print(f"\n【银行系统预测】该房产预估总价为: ${prediction[0][0]:.2f}")
# 6. 计算单价与区间报告 (对应项目书 3.1.3 产出物)
est_price = prediction[0][0]
area = 1500 # 我们假设的这套房面积

# 计算单价
price_per_sqft = est_price / area
avg_price_per_sqft = df['SalePrice'].sum() / df['GrLivArea'].sum() # 市场均价

# 计算偏差 (对应项目书 3.1.3 风险提示)
deviation = (price_per_sqft - avg_price_per_sqft) / avg_price_per_sqft * 100

# 模拟 95% 预测区间 (简化版：正负 10%)
lower_bound = est_price * 0.9
upper_bound = est_price * 1.1

print("\n" + "="*30)
print("     房 产 估 值 报 告 卡 片     ")
print("="*30)
print(f"预估总价：${est_price:,.2f}")
print(f"价格区间：${lower_bound:,.2f} - ${upper_bound:,.2f}")
print(f"预估单价：${price_per_sqft:.2f}/sqft")
print(f"市场均价：${avg_price_per_sqft:.2f}/sqft")
print(f"价格偏差：{deviation:+.2f}%")

# 风险预警 (对应项目书 59 行：偏差 > 15% 标红)
if abs(deviation) > 15:
    print("【风险提示】⚠️ 估值与市场均价偏差过大，触发人工复核！")
else:
    print("【系统结论】✅ 估值在合理区间内。")
print("="*30)