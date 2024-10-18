import sys, os, math, time
import matplotlib.pyplot as plt
from numpy import *


def sigmoid(x):
    return 1 / (1 + exp(-x))

# 双曲正切函数
def tanh1(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def hebbian(lr, w, x, d):
    x1 = [1, x[0], x[1]]
    net = sum([ww * xx for ww, xx in zip(w, x1)])
    o = tanh1(net) # 双曲正切函数
    w1 = [ww + lr * o * xx for ww, xx in zip(w, x1)]
    return w1

def perceptron(lr, w, x, d):
    x1 = [1, x[0], x[1]]
    net = sum([ww * xx for ww, xx in zip(w, x1)])
    o = 1 if net >= 0 else -1 # 二值函数
    w1 = [ww + lr * (d - o) * xx for ww, xx in zip(w, x1)]
    return w1

def delta(lr, w, x, d):
    x1 = [1, x[0], x[1]]
    net = sum([ww * xx for ww, xx in zip(w, x1)])
    o = tanh1(net)  # 双曲正切函数
    o1 = 1 - o**2   # 双曲正切函数的导数
    w1 = [ww + lr*(d - o) * o1 * xx for ww, xx in zip(w, x1)]
    return w1

def widrawhoff(lr, w, x, d):
    x1 = [1, x[0], x[1]]
    net = sum([ww * xx for ww, xx in zip(w, x1)])
    o = tanh1(net)  # 双曲正切函数
    w1 = [ww + lr * (d - o) * xx for ww, xx in zip(w, x1)]
    return w1

# def widrawhoff(lr, w, x, d):
#     x1 = [1, x[0], x[1]]
#     net = sum([ww * xx for ww, xx in zip(w, x1)])
#     o = net  # 不使用非线性激活函数
#     w1 = [ww + lr * (d - o) * xx for ww, xx in zip(w, x1)]
#     return w1

def correlation(lr, w, x, d):
    x1 = [1, x[0], x[1]]
    w1 = [ww + lr * d * xx for ww, xx in zip(w, x1)]  # 不使用非线性激活函数
    return w1


# 训练数据
xdim = [(-0.1, -0.2), (0.5, 0.5), (-0.5, 0.2), (-0.2, 0.5), (0.2, 0.1), (0.0, 0.8)]
ldim = [-1, 1, -1, -1, 1, 1]

# 权系数
wb = [0, 0, 0]  # [b, w1, w2]初始化为0

# 学习率
lr = 0.5

# 记录权重变化
w1_values = []
w2_values = []

print("使用的算法是：perceptron")
print('学习速率设置为：',lr)
# 进行两轮训练
for epoch in range(2):  # 两轮训练
    print(f"Epoch {epoch + 1}")
    for x, d in zip(xdim, ldim):
        # 在这里可以选择使用不同的算法，比如 hebbian, perceptron, delta, widrowhoff, correlation
        wb = perceptron(lr, wb, x, d)
        print(f"Sample {x}: Updated weights: {wb}")

        # 记录每次更新后的 w1 和 w2 值
        w1_values.append(wb[1])  # 记录 w1
        w2_values.append(wb[2])  # 记录 w2

# 输出最终的权系数
print("最终的权系数:", wb)
print("\n")
# 绘制 w1 和 w2 的空间位置
plt.figure()
plt.scatter(w1_values, w2_values, c='blue', label='w1, w2 positions')
plt.plot(w1_values, w2_values, 'r--')  # 用红色虚线连接轨迹
# 为每个点添加标注 (index + 1)
for i in range(len(w1_values)):
    plt.annotate(f'{i+1}', (w1_values[i], w2_values[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel('w1')
plt.ylabel('w2')
plt.title('perceptron_Weight positions after two epochs')
plt.legend()
plt.grid(True)

# 保存图像为文件
plt.savefig('perceptron_weight_positions.png')  # 保存为PNG图片

# 显示图像
plt.show()