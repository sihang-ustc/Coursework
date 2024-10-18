import numpy as np
import matplotlib.pyplot as plt
import os

# Adaline更新规则，使用线性输出
def adaline(lr, w, x, d):
    x1 = np.insert(x, 0, 1)  # 插入偏置
    net = np.dot(w, x1)
    w += lr * (d - net) * x1  # 使用线性输出
    return w

# 预测函数
def predict(w, x):
    x1 = np.insert(x, 0, 1)
    net = np.dot(w, x1)
    return 1 if net >= 0 else -1

# 数据归一化处理
def normalize_data(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    return (X - X_mean) / X_std

# 训练Adaline算法
def train_adaline(xdim, ldim, lr, epochs):
    w = np.zeros(xdim.shape[1] + 1)  # 初始化权重 (含偏置)
    errors = []
    accuracies = []

    for epoch in range(epochs):
        total_error = 0
        correct_predictions = 0
        for x, d in zip(xdim, ldim):
            w = adaline(lr, w, x, d)
            prediction = predict(w, x)
            if prediction == d:
                correct_predictions += 1
            total_error += (d - prediction) ** 2

        accuracy = correct_predictions / len(ldim) * 100
        accuracies.append(accuracy)
        errors.append(total_error)
        print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.2f}%, Total Error: {total_error}")

    return w, errors, accuracies

# 数据生成函数
def species_generator(mu1, sigma1, mu2, sigma2, n_samples, target, seed):
    rand = np.random.RandomState(seed)
    f1 = rand.normal(mu1, sigma1, n_samples)
    f2 = rand.normal(mu2, sigma2, n_samples)
    X = np.array([f1, f2]).T
    y = np.full((n_samples,), target)
    return X, y

# 数据生成
n_samples = 50
X1, y1 = species_generator(9000, 800, 300, 20, n_samples, 1, seed=1)   # 信天翁数据
X2, y2 = species_generator(1000, 200, 100, 15, n_samples, -1, seed=2)  # 猫头鹰数据

# 合并数据
X = np.vstack((X1, X2))
y = np.hstack((y1, y2))

# 数据归一化
X = normalize_data(X)

# 训练Adaline
learning_rate = 1e-8  # 较低的学习率
#learning_rate = 10
epochs = 20

weights, error_history, accuracy_history = train_adaline(X, y, learning_rate, epochs)

# 创建保存路径
output_path = '/media/data4/liuyj/xjy/人工神经网络_hw1/10'
os.makedirs(output_path, exist_ok=True)

# 保存误差变化曲线
plt.figure()
plt.plot(range(1, epochs + 1), error_history, label="Training Error")
plt.title("Training Error over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Total Error")
plt.grid(True)
error_plot_path = os.path.join(output_path, 'error_curve.png')
plt.savefig(error_plot_path)
plt.close()

# 保存分类准确率变化曲线
plt.figure()
plt.plot(range(1, epochs + 1), accuracy_history, label="Accuracy", color='green')
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.grid(True)
accuracy_plot_path = os.path.join(output_path, 'accuracy_curve.png')
plt.savefig(accuracy_plot_path)
plt.close()

# 绘制决策边界
def plot_decision_boundary(w, X, y, save_path):
    plt.figure()

    x_vals = np.linspace(min(X[:, 0]), max(X[:, 0]), 1000)
    y_vals = -(w[1] * x_vals + w[0]) / w[2]
    plt.plot(x_vals, y_vals, label="Decision Boundary")

    for i, (x, label) in enumerate(zip(X, y)):
        color = 'red' if label == 1 else 'blue'
        plt.scatter(x[0], x[1], color=color)

    plt.xlabel("Weight (g)")
    plt.ylabel("Wingspan (cm)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# 绘制分类边界并保存
boundary_plot_path = os.path.join(output_path, 'decision_boundary.png')
plot_decision_boundary(weights, X, y, boundary_plot_path)

# 输出生成的文件路径
print("Error curve saved at:", error_plot_path)
print("Accuracy curve saved at:", accuracy_plot_path)
print("Decision boundary saved at:", boundary_plot_path)
