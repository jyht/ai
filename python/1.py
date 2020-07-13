import numpy as np
from sklearn.linear_model import LogisticRegression
 
X_train = np.array([[1,2,3],[4,5,6],[10,9,8]])
Y_train = np.array([0,0,1])
 
X_test = np.array([[2,3,4],[9,8,7]])
Y_test = np.array([0,1])
 
# 逻辑回归模型
LR = LogisticRegression()
# 训练模型
LR.fit(X_train,Y_train)
 
print('预测结果:\n', LR.predict(X_test))
print('预测各标签的概率值:\n', LR.predict_proba(X_test))
 
print('训练集准确率:', LR.score(X_train,Y_train))

