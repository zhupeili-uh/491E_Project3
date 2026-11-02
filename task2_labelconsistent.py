import sys
sys.path.append('./BackdoorBox')

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage
import core

def main():
    # 1. 准备 CIFAR-10 数据集
    datasets_root_dir = './data'
    dataset = torchvision.datasets.CIFAR10

    # 训练集和测试集的数据预处理
    transform_train = Compose([
        ToPILImage(),
        RandomHorizontalFlip(),
        ToTensor()
    ])
    trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

    transform_test = Compose([
        ToTensor()
    ])
    testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
    
    # 【关键修改】：强行截取前 5000 张图片和标签，绕过 BackdoorBox 的类型检查
    trainset.data = trainset.data[:5000]
    trainset.targets = trainset.targets[:5000]

    # 2. 设置 Trigger (暗号/触发器)
    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[-3:, -3:] = 1.0

    # 3. 初始化 LabelConsistent 攻击器 (Clean Label Attack)
    # 注意：LabelConsistent 攻击通过添加对抗扰动来欺骗模型，因此需要额外传入 adv_model 和 adv_dataset_dir
    label_consistent = core.LabelConsistent(
        train_dataset=trainset,
        test_dataset=testset,
        model=core.models.ResNet(18),
        adv_model=core.models.ResNet(18),       # 新增：用于生成对抗样本的辅助模型
        adv_dataset_dir='./data/adv_dataset',   # 新增：生成的对抗样本的存放路径
        loss=nn.CrossEntropyLoss(),
        y_target=1,                             # 目标标签 (干净标签：投毒图像原本的真实标签也是1)
        poisoned_rate=0.05,
        pattern=pattern,
        weight=weight,
        seed=666,
        deterministic=True
    )

    # 4. 配置训练参数 (Schedule)
    schedule = {
        'device': 'mps',          
        'benign_training': False, 
        'batch_size': 64,        
        'num_workers': 2,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [20, 30],
        'epochs': 40,            
        'log_iteration_interval': 50,
        'test_epoch_interval': 5,
        'save_epoch_interval': 10,
        'save_dir': 'experiments',
        'experiment_name': 'ResNet-18_CIFAR-10_LabelConsistent' 
    }

    # 5. 执行训练和评估
    print("开始 LabelConsistent 干净标签投毒训练...")
    label_consistent.train(schedule)
    print("开始测试攻击成功率与干净数据准确率...")
    label_consistent.test(schedule)

if __name__ == "__main__":
    main()