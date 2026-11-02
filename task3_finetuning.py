import sys
import os
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.utils.prune as prune

sys.path.append('./BackdoorBox')
import core

# ================= 配置区 =================
# 目标标签，Task 1 和 Task 2 中均设置为 1
TARGET_LABEL = 1 
# ==========================================

def add_trigger(img):
    """添加 3x3 白块触发器"""
    img_poisoned = img.clone()
    img_poisoned[:, -3:, -3:] = 1.0  
    return img_poisoned

def evaluate(model, testloader, device, test_poisoned=False):
    """评估模型的 CDA (干净准确率) 或 ASR (攻击成功率)"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            if test_poisoned:
                # 【逻辑修复】跳过原本就是目标类别的图片 (标签为 1)
                valid_idx = labels != TARGET_LABEL
                inputs, labels = inputs[valid_idx], labels[valid_idx]
                if len(inputs) == 0: 
                    continue
                # 打上后门触发器，并将标签篡改为目标标签
                for i in range(len(inputs)):
                    inputs[i] = add_trigger(inputs[i])
                    labels[i] = TARGET_LABEL
                    
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total if total > 0 else 0

def main():
    if len(sys.argv) < 2:
        print("❌ 用法错误！请提供模型权重路径。")
        print("示例: python task3_defense.py ./experiments/ResNet-18_CIFAR-10_BadNets_xxx/ckpt_epoch_10.pth")
        sys.exit(1)
        
    model_path = sys.argv[1]
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🚀 使用设备: {device}")
    print(f"📂 正在读取模型: {model_path}")

    # 1. 准备数据集 (防御者只有少量干净数据，此处抽取 5% 作为验证/微调集)
    transform = transforms.Compose([transforms.ToTensor()])
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    defense_trainset = Subset(full_trainset, list(range(2500)))
    trainloader = DataLoader(defense_trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False)

    # 2. 实例化并加载中毒模型
    model = core.models.ResNet(18)
    if not os.path.exists(model_path):
        print(f"❌ 找不到权重文件: {model_path}")
        return
        
    state_dict = torch.load(model_path, map_location='cpu') # 先加载到CPU，防止不同设备间报错
    if 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    elif 'model' in state_dict:
        model.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict)

    model = model.to(device)
    
    # 3. 评估防御前基准值
    print("\n" + "="*40)
    print("      评估 [防御前] 模型性能")
    print("="*40)
    cda_before = evaluate(model, testloader, device, test_poisoned=False)
    asr_before = evaluate(model, testloader, device, test_poisoned=True)
    print(f"🔹 防御前 干净数据准确率 (CDA): {cda_before*100:.2f}%")
    print(f"🔺 防御前 攻击成功率 (ASR):   {asr_before*100:.2f}%")

    # 保存一个模型副本用于单独测试 Pruning 防御
    model_for_pruning = copy.deepcopy(model)

    # 4. 防御策略一：Fine-Tuning (微调)
    print("\n" + "="*40)
    print("      防御一: Fine-Tuning 过程")
    print("="*40)
    criterion = nn.CrossEntropyLoss()
    # 防御微调适当提高 epoch 数，确保洗掉后门特征
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    epochs = 8 
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}")

    cda_ft = evaluate(model, testloader, device, test_poisoned=False)
    asr_ft = evaluate(model, testloader, device, test_poisoned=True)
    print(f"🔹 微调后 干净数据准确率 (CDA): {cda_ft*100:.2f}%")
    print(f"🔻 微调后 攻击成功率 (ASR):   {asr_ft*100:.2f}%")

    # 5. 防御策略二：Pruning (剪枝)
    print("\n" + "="*40)
    print("      防御二: Pruning 过程")
    print("="*40)
    # 这里采用 L1 范数进行全局非结构化剪枝，剪掉激活绝对值最小的 40% 卷积层权重
    parameters_to_prune = []
    for module in model_for_pruning.modules():
        if isinstance(module, nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.40,
    )
    
    # 将 mask 永久应用到权重上
    for module, name in parameters_to_prune:
        prune.remove(module, name)

    cda_prune = evaluate(model_for_pruning, testloader, device, test_poisoned=False)
    asr_prune = evaluate(model_for_pruning, testloader, device, test_poisoned=True)
    print(f"🔹 剪枝后 干净数据准确率 (CDA): {cda_prune*100:.2f}%")
    print(f"🔻 剪枝后 攻击成功率 (ASR):   {asr_prune*100:.2f}%")
    
    # 6. 保存安全模型
    dir_name = os.path.basename(os.path.dirname(model_path))
    ft_save_path = f'./experiments/{dir_name}_finetuned.pth'
    pr_save_path = f'./experiments/{dir_name}_pruned.pth'
    torch.save(model.state_dict(), ft_save_path)
    torch.save(model_for_pruning.state_dict(), pr_save_path)
    print(f"\n✅ 修复完成！\n微调模型保存至: {ft_save_path}\n剪枝模型保存至: {pr_save_path}")

if __name__ == '__main__':
    main()