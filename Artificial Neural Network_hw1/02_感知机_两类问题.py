import numpy as np
import matplotlib.pyplot as plt

# 感知机算法核心代码
def perceptron(lr, w, x, d):
    x1 = [1, x[0], x[1]]  # 插入偏置项
    net = sum([ww * xx for ww, xx in zip(w, x1)])
    o = 1 if net >= 0 else -1 # 二值函数
    w1 = [ww + lr * (d - o) * xx for ww, xx in zip(w, x1)] # 更新权重
    return w1

# 预测函数
def predict(w, x):
    x1 = [1, x[0], x[1]]  # 插入偏置项
    net = sum([ww * xx for ww, xx in zip(w, x1)])  # 计算净输入
    return 1 if net >= 0 else -1  # 二值函数

# 绘制分类边界
def plot_decision_boundary(w, label):
    x_vals = np.linspace(-1, 1, 100)
    y_vals = -(w[1] * x_vals + w[0]) / w[2]  # 分类边界方程：w1*x1 + w2*x2 + b = 0
    plt.plot(x_vals, y_vals, label=f'{label} (w1={w[1]:.2f}, w2={w[2]:.2f}, b={w[0]:.2f})')

# 训练感知机算法
def train_perceptron(lr, epochs):
    # 初始化权重
    w = [0, 0, 0]  # [b, w1, w2]

    # 样本数据
    xdim = [(-0.1, -0.2), (0.5, 0.5), (-0.5, 0.2), (-0.2, 0.5), (0.2, 0.1), (0.0, 0.8)]
    ldim = [-1, 1, -1, -1, 1, 1]

    for epoch in range(epochs):
        correct_predictions = 0
        for x, d in zip(xdim, ldim):
            w = perceptron(lr, w, x, d)  # 更新权重
            prediction = predict(w, x)  # 预测
            if prediction == d:
                correct_predictions += 1

        # 输出每轮的分类准确率
        accuracy = correct_predictions / len(ldim) * 100
        print(f'Epoch {epoch+1}, Learning Rate {lr}, Accuracy: {accuracy:.2f}%')

    return w

# 比较不同学习速率的训练结果
def compare_learning_rates():
    # 学习率
    learning_rates = [0.1]
    epochs = 10  # 迭代次数

    plt.figure()
    
    for lr in learning_rates:
        # 训练感知机算法
        final_weights = train_perceptron(lr, epochs)
        # 绘制分类边界
        plot_decision_boundary(final_weights, f'LR = {lr}')

    # 绘制散点图（样本数据）
    xdim = [(-0.1, -0.2), (0.5, 0.5), (-0.5, 0.2), (-0.2, 0.5), (0.2, 0.1), (0.0, 0.8)]
    ldim = [-1, 1, -1, -1, 1, 1]

    for i, (x, label) in enumerate(zip(xdim, ldim)):
        color = 'red' if label == 1 else 'blue'
        plt.scatter(x[0], x[1], color=color)
    
    # 添加图例和标签
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title("Decision Boundary with Different Learning Rates")
    plt.grid(True)
    plt.savefig(f'感知机_decision_boundary{lr}.png')  # 保存为PNG图片
    plt.show()

compare_learning_rates()
