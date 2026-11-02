import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def main():
    adv_dir = './data/adv_dataset'
    
    # 检查对抗样本文件夹是否已经生成
    if not os.path.exists(adv_dir):
        print(f"错误: 找不到 {adv_dir} 文件夹。")
        print("请确认 task2_labelconsistent.py 的前置进度条已经跑完！")
        return

    # 1. 加载生成好的对抗样本 (带有肉眼难以察觉的对抗噪声)
    try:
        adv_dataset = datasets.ImageFolder(adv_dir, transform=transforms.ToTensor())
        adv_img, original_label = adv_dataset[0] # 取出第一张图
    except Exception as e:
        print("读取图像失败:", e)
        return

    # 2. 准备后门 Trigger (右下角 3x3 白块)
    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[-3:, -3:] = 1.0

    # 3. 将 Trigger 盖在对抗样本上，合成最终的“中毒图像”
    poisoned_img = adv_img.clone()
    pattern_tensor = (pattern / 255.0).unsqueeze(0).expand(3, 32, 32)
    weight_tensor = weight.unsqueeze(0).expand(3, 32, 32)
    
    poisoned_img = (1 - weight_tensor) * poisoned_img + weight_tensor * pattern_tensor

    # 4. 画图对比并保存
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # 左图：只带对抗噪声的图片 (看起来标签依然是干净的)
    axes[0].imshow(adv_img.permute(1, 2, 0))
    axes[0].set_title(f"Adv Image (Label: {original_label})")
    axes[0].axis('off')

    # 右图：带对抗噪声 + 白块 Trigger 的最终投毒图
    axes[1].imshow(poisoned_img.permute(1, 2, 0))
    axes[1].set_title(f"Poisoned Image w/ Trigger")
    axes[1].axis('off')

    plt.tight_layout()
    # 将图片保存下来，插入报告
    plt.savefig('task2_poisoned_sample.png', dpi=300, bbox_inches='tight')
    print("图像已保存为 task2_poisoned_sample.png")
    plt.show()

if __name__ == "__main__":
    main()