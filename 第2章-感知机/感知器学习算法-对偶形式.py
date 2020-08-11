'''
    《统计学习方法》 第二章 感知机学习算法--对偶形式
    author: geds
    time  : 2020-08-11
'''
import random
import math
import matplotlib.pyplot as plt
import numpy as np


# 在一个圆形区域内随机生成数据
# center: 2D圆形中心; radius: 半径; num: 数据量; seed: 随机数种子; label: 数据标签;
def data_generate_in_circle(center, radius, num, seed=10, label=1):
    sample_x1 = []
    sample_x2 = []
    sample_y = []
    random.seed(seed)
    for i in range(int(num)):
        angle = random.randint(0, 360)
        distant = random.uniform(0, radius)
        x = center[0] + distant / ((1 + math.fabs(math.tan(angle))) ** 0.5)
        y = center[1] + distant * math.tan(angle) / ((1 + math.fabs(math.tan(angle))) ** 0.5)
        sample_x1.append(x)
        sample_x2.append(y)
        sample_y.append(label)
    return sample_x1, sample_x2, sample_y


class1_x1, class1_x2, class1_y = data_generate_in_circle([1, 1], 3, 20, seed=5, label=1)  # positive
class2_x1, class2_x2, class2_y = data_generate_in_circle([5, 5], 3, 20, seed=7, label=-1)  # negative
# 样本集
X1 = np.array(class1_x1 + class2_x1)  # list类型转换为array类型
X2 = np.array(class1_x2 + class2_x2)
X = np.array([X1, X2])                # 向量组合成矩阵
Y = np.array(class1_y + class2_y)
# 初始化参数
N = X.shape[1]                        # 矩阵X的第二个维度
w = np.array([0.0, 0.0])
b = 0.0
alpha = np.zeros(N)
eta = 0.2  # learning rate
# 定义相关变量
false_num = 0
true_num = 0
task_over = False
x = np.array([0, 0])  # 一个实例输入
x_temp = np.array([0.0, 0.0])
G = np.zeros((N, N))
# 计算Gram矩阵, 用于存储实例x_i 和实例 x_j的点积
for i in range(N):
    for j in range(N):
        G[i][j] = np.dot(X[:, i], np.array(X[:, j]))

while not task_over:
    true_num = 0
    for i in range(N):
        while Y[i] * (np.dot(Y * G[:, i], alpha) + b) <= 0:
            alpha[i] += eta
            b += Y[i] * eta
            false_num += 1
            true_num -= 1
        true_num += 1
    if true_num == N:
        task_over = True
w = np.dot(Y * X, alpha)
b = np.dot(Y, alpha)
print("模型参数: w:{}, b:{}\n总共误分类次数:{}".format(w, b, false_num))

# plot figure
line_x_ = np.linspace(4, 5, 20)
line_y_ = -(w[0] * line_x_ + b) / (w[1] + 0.001)
plt.figure(figsize=(4, 3))
plt.plot(line_x_, line_y_, c='c')
plt.scatter(class1_x1, class1_x2, s=30, c='r', marker='o', alpha=0.6)
plt.scatter(class2_x1, class2_x2, s=30, c='b', marker='o', alpha=0.6)
plt.show()