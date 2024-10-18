import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2.ximgproc import guidedFilter
import os
import scipy.sparse
import scipy.sparse.linalg

# 用于显示图像
def show_image(image, title="Image"):
    image = cv2.convertScaleAbs(image)  # 转换为 8 位无符号整数
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# 保存图像的函数
def save_image(image, filename, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, image)
    print(f"Image saved at: {save_path}")

# 计算输入带雾图像的暗通道
def get_dark_channel(image, window_size=15):
    min_channel = np.amin(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

# 替换最小值滤波顺序的代码（求暗通道）
# def get_dark_channel(image, window_size=15):
#     print(image.shape)
    
#     # 分离 RGB 通道
#     b, g, r = cv2.split(image)
#     print(b.shape)
    
#     # 创建结构元素
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    
#     # 对每个通道进行腐蚀操作以获取领域内最小值
#     min_b = cv2.erode(b, kernel)
#     min_g = cv2.erode(g, kernel)
#     min_r = cv2.erode(r, kernel)

#     merged_image = cv2.merge((min_b,min_g,min_r))
#     print(merged_image.shape)
#     min_channel = np.amin(merged_image, axis=2)
#     dark_channel = min_channel
    
#     return dark_channel

# 估计大气亮度
def get_atmospheric_light(image, dark_channel):
    num_pixels = image.shape[0] * image.shape[1]
    num_brightest = int(max(num_pixels * 0.001, 1))  # 至少大于1
    
    dark_channel_flat = dark_channel.ravel()
    indices = np.argsort(dark_channel_flat)[-num_brightest:]
    
    brightest = np.unravel_index(indices, dark_channel.shape)
    atmospheric_light = np.mean(image[brightest], axis=0)
    
    return atmospheric_light

# 计算传导图像
def get_transmission(image, atmospheric_light, window_size=15, omega=0.95):
    atmospheric_light = atmospheric_light.reshape(1, 1, 3)
    norm_image = image / atmospheric_light
    transmission = 1 - omega * get_dark_channel(norm_image, window_size)
    return transmission

# 计算拉普拉斯矩阵
def get_laplacian(image, epsilon=1e-7, window_size=3):
    h, w, c = image.shape
    num_pixels = h * w
    laplacian = scipy.sparse.lil_matrix((num_pixels, num_pixels))

    # 用颜色相似性来计算拉普拉斯矩阵
    for y in range(h):
        for x in range(w):
            index = y * w + x
            window_y_min = max(0, y - window_size // 2)
            window_y_max = min(h, y + window_size // 2 + 1)
            window_x_min = max(0, x - window_size // 2)
            window_x_max = min(w, x + window_size // 2 + 1)

            window = image[window_y_min:window_y_max, window_x_min:window_x_max]
            center_pixel = image[y, x]
            
            # 计算每个通道的方差
            variance = np.var(window, axis=(0, 1), ddof=1) + epsilon  # 计算每个通道的方差
            variance = variance[np.newaxis, np.newaxis, :]  # 调整形状为 (1, 1, 3) 以匹配

            # 使用各个通道的方差进行计算
            weights = np.exp(-np.sum((window - center_pixel[np.newaxis, np.newaxis, :]) ** 2 / variance, axis=2))
            weights /= np.sum(weights)

            for wy in range(window.shape[0]):
                for wx in range(window.shape[1]):
                    neighbor_index = (window_y_min + wy) * w + (window_x_min + wx)
                    laplacian[index, neighbor_index] -= weights[wy, wx]
                    laplacian[index, index] += weights[wy, wx]

    return laplacian.tocsr()



# Soft Matting 方法优化传导图像
def soft_matting(image, transmission, laplacian, lambd=1e-4):  #1e-4
    h, w = transmission.shape
    num_pixels = h * w
    
    # 将传导图像展平为一维向量
    t_flat = transmission.flatten()

    # 构建线性方程 (I + λL) t = t_0
    identity = scipy.sparse.eye(num_pixels)
    A = identity + lambd * laplacian
    b = t_flat

    # 使用共轭梯度法求解线性方程
    t_optimized, _ = scipy.sparse.linalg.cg(A, b)
    
    # 将优化后的 t 重新变为二维图像
    t_optimized = t_optimized.reshape((h, w))
    
    return t_optimized
 
# 引导滤波优化传导图
def refine_transmission(image, transmission, r=60, eps=1e-4):
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # 将 transmission 转换为浮点型 CV_32F，避免数据类型错误
    transmission = transmission.astype(np.float32)

    # 应用引导滤波进行优化
    refined_transmission = guidedFilter(gray, transmission, r, eps)
    return refined_transmission

# 恢复图像
def recover_image(image, transmission, atmospheric_light, t0=0.1):
    transmission = np.clip(transmission, t0, 1)
    recovered = np.zeros_like(image, dtype=np.float32)

    for channel in range(image.shape[2]):
        recovered[..., channel] = (image[..., channel] - atmospheric_light[channel]) / transmission + atmospheric_light[channel]

    recovered = np.clip(recovered, 0, 255).astype(np.uint8)
    return recovered

# 去雾流程
def dehaze(image_path, output_dir):
    # 读取带雾图像
    image = cv2.imread(image_path)
    image = image.astype(np.float64)
    
    # 1. 计算暗通道
    dark_channel = get_dark_channel(image)
    
    # 2. 估计大气亮度
    atmospheric_light = get_atmospheric_light(image, dark_channel)
    
    # 3. 计算传导图像
    transmission = get_transmission(image, atmospheric_light)
    
    # 4. 计算拉普拉斯矩阵
    laplacian = get_laplacian(image)
    
    # 5. 使用 Soft Matting 优化传导图像
    optimized_transmission = soft_matting(image, transmission, laplacian)

    # 6. 使用引导滤波进一步优化传导图像
    refined_transmission = refine_transmission(image, optimized_transmission)
    
    # 6. 恢复原始场景图像
    recovered_image = recover_image(image, refined_transmission, atmospheric_light)
    
    # 保存结果图像到指定路径
    save_image(cv2.convertScaleAbs(image), "original_image.jpg", output_dir)
    save_image(cv2.convertScaleAbs(dark_channel), "dark_channel.jpg", output_dir)
    save_image(cv2.convertScaleAbs(optimized_transmission * 255), "optimized_transmission_map.jpg", output_dir)
    save_image(cv2.convertScaleAbs(refined_transmission * 255), "refined_transmission_map.jpg", output_dir)
    save_image(recovered_image, "recovered_image.jpg", output_dir)

    # 可视化显示
    show_image(image, "Original Haze Image")
    show_image(dark_channel, "Dark Channel")
    show_image(optimized_transmission, "Optimized Transmission Map")
    show_image(refined_transmission, "Refined Transmission Map")
    show_image(recovered_image, "Recovered Image")

# 执行去雾流程，并保存结果到指定路径
output_directory = "/media/data4/liuyj/xjy/Experiment-1/实验结果/002"
dehaze("/media/data4/liuyj/xjy/Experiment-1/rainy2.png", output_directory)
