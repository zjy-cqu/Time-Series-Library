import torch
import numpy as np
import matplotlib.pyplot as plt
import os
# 读取文件
A = np.load('A.npy')
print("Loaded tensor shape:", A.shape)

# 创建目录以保存所有图像
output_dir = "draw"
os.makedirs(output_dir, exist_ok=True)

# 遍历所有批次和头
B, H, L, S = A.shape
for b in range(B):  # 遍历 batch
    for h in range(H):  # 遍历 head
        # 获取当前切片
        slice_to_plot = A[b, h, :, :]  # 形状 [L, S]

        # 绘制热力图
        plt.figure(figsize=(10, 8))
        plt.title(f"Tensor Slice (Batch={b}, Head={h})")
        plt.imshow(slice_to_plot, cmap='viridis', aspect='auto')
        plt.colorbar(label='Value')
        plt.xlabel('Dimension S')
        plt.ylabel('Dimension L')

        # 保存图像
        filename = os.path.join(output_dir, f"tensor_b{b}_h{h}.png")
        plt.savefig(filename)
        plt.close()  # 关闭图像以释放内存

print(f"All visualizations are saved in '{output_dir}'")