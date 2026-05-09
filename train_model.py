import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# 加载数据
data = pd.read_csv('training_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 训练线性回归模型
model = LinearRegression()
model.fit(X, y)

# 保存模型二进制文件
joblib.dump(model, 'linear_model.pkl')

# 将模型系数保存为文本文件，方便查看
with open('linear_model.txt', 'w') as f:
    f.write(f'Coefficients: {model.coef_}\nIntercept: {model.intercept_}\n')