import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 设置随机种子
np.random.seed(42)

# ======================================
# 1. 生成简单的泰坦尼克号数据集
# ======================================
print("生成泰坦尼克号数据集...")

def generate_data(n_samples=891):
    """生成简单的泰坦尼克号数据集"""
    data = pd.DataFrame()
    
    # PassengerId
    data['PassengerId'] = range(1, n_samples + 1)
    
    # Pclass: 舱位等级 (1, 2, 3)
    data['Pclass'] = np.random.choice([1, 2, 3], size=n_samples, p=[0.24, 0.20, 0.56])
    
    # Sex: 性别
    data['Sex'] = np.random.choice(['female', 'male'], size=n_samples, p=[0.35, 0.65])
    
    # Age: 年龄
    data['Age'] = np.random.randint(0, 80, size=n_samples)
    
    # Fare: 票价
    fares = []
    for pclass in data['Pclass']:
        if pclass == 1:
            fares.append(round(np.random.uniform(50, 500), 2))
        elif pclass == 2:
            fares.append(round(np.random.uniform(15, 75), 2))
        else:
            fares.append(round(np.random.uniform(0, 30), 2))
    data['Fare'] = fares
    
    # Survived: 存活状态（模拟真实情况）
    survived = []
    for i in range(n_samples):
        sex = data['Sex'][i]
        pclass = data['Pclass'][i]
        
        # 基础存活率：女性约70%，男性约20%
        if sex == 'female':
            prob = 0.70
        else:
            prob = 0.20
        
        # 舱位调整
        if pclass == 1:
            prob *= 1.2
        elif pclass == 3:
            prob *= 0.8
        
        prob = min(1.0, max(0.0, prob))
        survived.append(1 if np.random.random() < prob else 0)
    data['Survived'] = survived
    
    return data

# 生成数据集
data = generate_data(891)
print(f"数据集生成完成，共 {len(data)} 条记录")

# ======================================
# 2. 简单划分数据集（70%训练，30%测试）
# ======================================
print("\n" + "="*50)
print("划分数据集（70%训练，30%测试）")
print("="*50)

# 特征和标签
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

# 转换性别为数值
X['Sex'] = X['Sex'].map({'female': 0, 'male': 1})

# 划分训练集和测试集（70-30分割）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"训练集大小: {len(X_train)} ({len(X_train)/len(data):.0%})")
print(f"测试集大小: {len(X_test)} ({len(X_test)/len(data):.0%})")

# ======================================
# 3. 训练模型
# ======================================
print("\n" + "="*50)
print("训练模型")
print("="*50)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ======================================
# 4. 模型评估
# ======================================
print("\n" + "="*50)
print("模型评估")
print("="*50)

y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"训练集准确率: {train_accuracy:.4f}")

y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"测试集准确率: {test_accuracy:.4f}")

# ======================================
# 5. 保存结果
# ======================================
print("\n" + "="*50)
print("保存结果")
print("="*50)

# 保存训练集和测试集
X_train['Survived'] = y_train
X_test['Survived'] = y_test

X_train.to_csv('train_set.csv', index=False)
X_test.to_csv('test_set.csv', index=False)

print("训练集已保存: train_set.csv")
print("测试集已保存: test_set.csv")

print("\n项目完成！")