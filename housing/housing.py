import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 导入数据
data = pd.read_csv('housing.csv')
print(data.info())  # 查看数据类型，是否有缺失值
print(data.describe())  # 统计信息

# 观察分布
data.hist(bins=50, figsize=(20,15))
plt.show()

# 预处理：检查缺失值
print(data.isnull().sum())
# 填充缺失值
data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].median())

# 处理分类特征
data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)
# 特征和目标
X = data.drop('median_house_value', axis=1)  # 目标是median_house_value
y = data['median_house_value']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 线性回归模型
model = LinearRegression()
model.fit(X_train_scaled, y_train)
# 预测
y_pred = model.predict(X_test_scaled)
# 评估
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
# 原始散点图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices (Scatter Plot)')

# 线性模型直线
plt.subplot(1, 2, 2)
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='Linear Model')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices (with Linear Model)')
plt.legend()
plt.tight_layout()
plt.show()