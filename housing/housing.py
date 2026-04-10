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
# 可视化：预测 vs 实际
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

# 闭合形式参数求解
# 将数据转换为numpy数组
X_train_np = X_train_scaled.astype(float)
X_test_np = X_test_scaled.astype(float)
y_train_np = y_train.values
y_test_np = y_test.values

# 添加截距项（一列全1）
X_train_int = np.c_[np.ones(X_train_np.shape[0]), X_train_np]
X_test_int = np.c_[np.ones(X_test_np.shape[0]), X_test_np]

# 闭合形式求解参数
theta_closed = np.linalg.inv(X_train_int.T @ X_train_int) @ X_train_int.T @ y_train_np
y_pred_closed = X_test_int @ theta_closed
mse_closed = mean_squared_error(y_test_np, y_pred_closed)
print(f'Closed-form MSE: {mse_closed}')

# 梯度下降参数优化
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1 / m) * X.T @ errors
        theta -= alpha * gradient
    return theta

# 初始化theta
theta_gd = np.zeros(X_train_int.shape[1])
# 学习率和迭代次数
alpha = 0.01
iterations = 1000
# 运行梯度下降
theta_gd = gradient_descent(X_train_int, y_train_np, theta_gd, alpha, iterations)
y_pred_gd = X_test_int @ theta_gd
mse_gd = mean_squared_error(y_test_np, y_pred_gd)
print(f'Gradient Descent MSE: {mse_gd}')