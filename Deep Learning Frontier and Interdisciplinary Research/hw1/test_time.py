import torch
import time

# 矩阵的尺寸
A = torch.randn(2**10, 2**16)
B = torch.randn(2**16, 2**5)
C = torch.randn(2**5, 2**16)

# 矩阵乘法性能测试
def matrix_multiply_test(A, B, C, device_name):
    # 将矩阵移动到相应设备
    A = A.to(device_name)
    B = B.to(device_name)
    C = C.to(device_name)

    # 测试 AB
    start_time = time.time()
    result_AB = torch.matmul(A, B)
    end_time = time.time()
    print(f"AB 计算时间 ({device_name}): {end_time - start_time} 秒")
    
    # 测试 AC^T
    start_time = time.time()
    result_AC_t = torch.matmul(A, C.t())
    end_time = time.time()
    print(f"AC^T 计算时间 ({device_name}): {end_time - start_time} 秒")
    
    # 使用 B.T 初始化 C
    C_shared = B.t()

    # 测试 AC^T 使用 B^T 初始化
    start_time = time.time()
    result_AC_t_shared = torch.matmul(A, C_shared)
    end_time = time.time()
    print(f"AC^T (使用 B^T 初始化) 计算时间 ({device_name}): {end_time - start_time} 秒")

# 在 CPU 上运行
print("在 CPU 上测试")
matrix_multiply_test(A, B, C, torch.device('cpu'))

# 检查是否有 GPU 可用，并在 GPU 上运行
if torch.cuda.is_available():
    print("\n在 GPU 上测试")
    matrix_multiply_test(A, B, C, torch.device('cuda'))
else:
    print("\n未检测到 GPU，跳过 GPU 测试")
