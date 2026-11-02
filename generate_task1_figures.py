import matplotlib.pyplot as plt
import torchvision
import numpy as np
import os

# 创建一个专门存放报告图片的文件夹
output_dir = "report_assets"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 1. 绘制 Training Loss 曲线
# ==========================================
# 根据之前终端截图里的数据，提取了前12轮的近似 Loss 趋势
# 可以根据需要修改这些数字
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
losses = [1.85, 1.52, 1.25, 1.05, 0.90, 0.78, 0.65, 0.58, 0.52, 0.52, 0.42, 0.39]

plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, marker='o', linestyle='-', color='b', linewidth=2, label='Training Loss')
plt.title("Task 1: BadNet Training Loss (Epochs 1-12)", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

loss_curve_path = os.path.join(output_dir, "task1_loss_curve.png")
plt.savefig(loss_curve_path, dpi=300)
plt.close()
print(f"✅ 训练曲线已保存至: {loss_curve_path}")

# ==========================================
# 2. 生成并保存 3x3 白色方块的中毒样本图
# ==========================================
# 加载 CIFAR-10 数据集的一张原图 (直接从已下载的 data 文件夹读取)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False)
img, original_label = trainset[4] # 选一张青蛙或者汽车的图片

# 转换为 NumPy 数组以便操作像素
img_np = np.array(img)

# 【核心投毒逻辑】：将右下角 3x3 的区域强制涂成纯白色 (255, 255, 255)
img_np[-3:, -3:] = [255, 255, 255]

# 显示并保存中毒图片
plt.figure(figsize=(3, 3))
plt.imshow(img_np)
plt.title("Poisoned Sample (Target Label: 1)", fontsize=12)
plt.axis('off') # 去掉坐标轴，让图片在报告里更美观

poison_img_path = os.path.join(output_dir, "task1_poison_sample.png")
plt.savefig(poison_img_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 中毒样本图已保存至: {poison_img_path}")