import sys
sys.path.append('./BackdoorBox')

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip
import core

def main():
    # 1. 准备 CIFAR-10 数据集
    datasets_root_dir = './data'
    dataset = torchvision.datasets.CIFAR10

    # 训练集和测试集的数据预处理
    transform_train = Compose([
        RandomHorizontalFlip(),
        ToTensor()
    ])
    trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

    transform_test = Compose([
        ToTensor()
    ])
    testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

    # 2. 设置 BadNet 的 Trigger (暗号/触发器)
    # 官方示例在图片右下角生成了一个 3x3 的白色方块 (像素值255) 作为后门特征
    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[-3:, -3:] = 1.0

    # 3. 初始化 BadNets 攻击器
    badnets = core.BadNets(
        train_dataset=trainset,
        test_dataset=testset,
        model=core.models.ResNet(18), # 调用工具箱内置的 ResNet18 模型
        loss=nn.CrossEntropyLoss(),
        y_target=1,                   # 目标错误标签 (脏标签)
        poisoned_rate=0.05,           # 投毒比例：5% 的训练数据会被打上后门
        pattern=pattern,
        weight=weight,
        seed=666,
        deterministic=True
    )

    # 4. 配置训练参数 (Schedule)
    schedule = {
        'device': 'mps',          # 若工具箱不支持 Mac MPS，则使用 CPU 训练
        'benign_training': False, # 设置为 False 表示进行带毒训练
        'batch_size': 512,        # 适度降低 batch_size 适配 Mac 内存
        'num_workers': 2,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [150, 180],
        'epochs': 200,            # 建议初次测试时改成 2，确认代码跑通后再改回 200
        'log_iteration_interval': 20,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,
        'save_dir': 'experiments',
        'experiment_name': 'ResNet-18_CIFAR-10_BadNets'
    }

    # 5. 执行训练和评估
    print("开始 BadNet 投毒训练...")
    badnets.train(schedule)
    print("开始测试攻击成功率与干净数据准确率...")
    badnets.test(schedule)

if __name__ == "__main__":
    main()