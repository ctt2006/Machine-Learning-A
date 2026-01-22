"""
训练脚本 - 手动实现优化器
"""
import os
import torch
import numpy as np
from model import CNN
from data_loader import create_data_loaders
import json


class SGD:
    """手动实现的SGD优化器"""
    def __init__(self, model, lr=0.001, momentum=0.9, weight_decay=0.0001):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}
        
        # 初始化速度
        for name, param in model.get_parameters():
            self.velocities[name] = torch.zeros_like(param)
    
    def step(self):
        """更新参数"""
        # 更新卷积层
        if self.model.conv1.weight_grad is not None:
            grad = self.model.conv1.weight_grad + self.weight_decay * self.model.conv1.weight
            self.velocities['conv1.weight'] = self.momentum * self.velocities['conv1.weight'] - self.lr * grad
            self.model.conv1.weight.data += self.velocities['conv1.weight']
        
        if self.model.conv1.bias_grad is not None:
            grad = self.model.conv1.bias_grad
            self.velocities['conv1.bias'] = self.momentum * self.velocities['conv1.bias'] - self.lr * grad
            self.model.conv1.bias.data += self.velocities['conv1.bias']
        
        if self.model.conv2.weight_grad is not None:
            grad = self.model.conv2.weight_grad + self.weight_decay * self.model.conv2.weight
            self.velocities['conv2.weight'] = self.momentum * self.velocities['conv2.weight'] - self.lr * grad
            self.model.conv2.weight.data += self.velocities['conv2.weight']
        
        if self.model.conv2.bias_grad is not None:
            grad = self.model.conv2.bias_grad
            self.velocities['conv2.bias'] = self.momentum * self.velocities['conv2.bias'] - self.lr * grad
            self.model.conv2.bias.data += self.velocities['conv2.bias']
        
        if self.model.conv3.weight_grad is not None:
            grad = self.model.conv3.weight_grad + self.weight_decay * self.model.conv3.weight
            self.velocities['conv3.weight'] = self.momentum * self.velocities['conv3.weight'] - self.lr * grad
            self.model.conv3.weight.data += self.velocities['conv3.weight']
        
        if self.model.conv3.bias_grad is not None:
            grad = self.model.conv3.bias_grad
            self.velocities['conv3.bias'] = self.momentum * self.velocities['conv3.bias'] - self.lr * grad
            self.model.conv3.bias.data += self.velocities['conv3.bias']
        
        # 更新全连接层
        if self.model.fc1.weight_grad is not None:
            grad = self.model.fc1.weight_grad + self.weight_decay * self.model.fc1.weight
            self.velocities['fc1.weight'] = self.momentum * self.velocities['fc1.weight'] - self.lr * grad
            self.model.fc1.weight.data += self.velocities['fc1.weight']
        
        if self.model.fc1.bias_grad is not None:
            grad = self.model.fc1.bias_grad
            self.velocities['fc1.bias'] = self.momentum * self.velocities['fc1.bias'] - self.lr * grad
            self.model.fc1.bias.data += self.velocities['fc1.bias']
        
        if self.model.fc2.weight_grad is not None:
            grad = self.model.fc2.weight_grad + self.weight_decay * self.model.fc2.weight
            self.velocities['fc2.weight'] = self.momentum * self.velocities['fc2.weight'] - self.lr * grad
            self.model.fc2.weight.data += self.velocities['fc2.weight']
        
        if self.model.fc2.bias_grad is not None:
            grad = self.model.fc2.bias_grad
            self.velocities['fc2.bias'] = self.momentum * self.velocities['fc2.bias'] - self.lr * grad
            self.model.fc2.bias.data += self.velocities['fc2.bias']
    
    def zero_grad(self):
        """清零梯度"""
        self.model.zero_grad()


def sigmoid(x):
    """Sigmoid函数"""
    # 防止溢出
    x = torch.clamp(x, min=-500, max=500)
    return 1.0 / (1.0 + torch.exp(-x))


def binary_cross_entropy_loss(pred, target):
    """
    二元交叉熵损失
    pred: (batch, 1) 未经过sigmoid的logits
    target: (batch,) 标签 (0或1)
    """
    pred_sigmoid = sigmoid(pred.squeeze())
    
    # 防止log(0)
    eps = 1e-8
    pred_sigmoid = torch.clamp(pred_sigmoid, min=eps, max=1-eps)
    
    loss = -target * torch.log(pred_sigmoid) - (1 - target) * torch.log(1 - pred_sigmoid)
    return torch.mean(loss)


def binary_cross_entropy_loss_backward(pred, target):
    """
    二元交叉熵损失的梯度
    """
    pred_sigmoid = sigmoid(pred.squeeze())
    batch_size = pred.shape[0]
    
    # 梯度计算
    grad = (pred_sigmoid - target) / batch_size
    return grad.unsqueeze(1)


def train_epoch(model, train_loader, optimizer, device='cpu'):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, labels, _) in enumerate(train_loader):
        # 移动到设备
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model.forward(images)
        
        # 计算损失
        loss = binary_cross_entropy_loss(outputs, labels)
        total_loss += loss.item()
        
        # 反向传播
        model.zero_grad()
        grad_output = binary_cross_entropy_loss_backward(outputs, labels)
        model.backward(grad_output)
        
        # 更新参数
        optimizer.step()
        optimizer.zero_grad()
        
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    return total_loss / num_batches


def evaluate(model, test_loader, device='cpu'):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_filenames = []
    
    with torch.no_grad():
        for images, labels, filenames in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model.forward(images)
            preds = sigmoid(outputs.squeeze()) > 0.5
            
            all_preds.extend(preds.cpu().numpy().astype(int).tolist())
            all_labels.extend(labels.cpu().numpy().astype(int).tolist())
            all_filenames.extend(filenames)
    
    return all_preds, all_labels, all_filenames


def calculate_metrics(preds, labels):
    """计算评估指标"""
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    # 对于有缺陷类别（标签为1）的指标
    precision = precision_score(labels, preds, pos_label=1, zero_division=0)
    recall = recall_score(labels, preds, pos_label=1, zero_division=0)
    f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    accuracy = accuracy_score(labels, preds)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy
    }


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 数据路径
    dataset_dir = 'dataset'
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    
    # 检查数据集是否存在
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("请先解压dataset.zip到dataset目录")
        return
    
    # 超参数
    batch_size = 16  # 减小batch size以适应手动实现
    learning_rate = 0.001
    num_epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print(f"训练轮数: {num_epochs}")
    
    # 创建数据加载器
    print("加载数据...")
    train_loader, test_loader, test_dataset = create_data_loaders(train_dir, test_dir, batch_size)
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"测试样本数: {len(test_loader.dataset)}")
    
    # 创建模型
    print("创建模型...")
    model = CNN()
    if device == 'cuda':
        # 将模型参数移到GPU
        model.conv1.weight = model.conv1.weight.cuda()
        model.conv1.bias = model.conv1.bias.cuda()
        model.conv2.weight = model.conv2.weight.cuda()
        model.conv2.bias = model.conv2.bias.cuda()
        model.conv3.weight = model.conv3.weight.cuda()
        model.conv3.bias = model.conv3.bias.cuda()
        model.fc1.weight = model.fc1.weight.cuda()
        model.fc1.bias = model.fc1.bias.cuda()
        model.fc2.weight = model.fc2.weight.cuda()
        model.fc2.bias = model.fc2.bias.cuda()
    
    # 创建优化器
    optimizer = SGD(model, lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"平均训练损失: {train_loss:.4f}")
        
        # 每个epoch后评估
        if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
            print("评估模型...")
            preds, labels, filenames = evaluate(model, test_loader, device)
            metrics = calculate_metrics(preds, labels)
            print(f"测试集指标:")
            print(f"  Precision (defective): {metrics['precision']:.4f}")
            print(f"  Recall (defective): {metrics['recall']:.4f}")
            print(f"  F1-score (defective): {metrics['f1_score']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    # 最终评估并生成JSON输出
    print("\n生成最终预测结果...")
    preds, labels, filenames = evaluate(model, test_loader, device)
    metrics = calculate_metrics(preds, labels)
    
    print("\n最终测试集指标:")
    print(f"  Precision (defective): {metrics['precision']:.4f}")
    print(f"  Recall (defective): {metrics['recall']:.4f}")
    print(f"  F1-score (defective): {metrics['f1_score']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    # 生成JSON输出
    results = {}
    for filename, pred in zip(filenames, preds):
        results[filename] = bool(pred)  # True=有缺陷, False=无缺陷
    
    output_file = 'results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n预测结果已保存到: {output_file}")
    
    # 保存模型
    model_file = 'model.pth'
    model_state = {
        'conv1.weight': model.conv1.weight.cpu().clone(),
        'conv1.bias': model.conv1.bias.cpu().clone(),
        'conv2.weight': model.conv2.weight.cpu().clone(),
        'conv2.bias': model.conv2.bias.cpu().clone(),
        'conv3.weight': model.conv3.weight.cpu().clone(),
        'conv3.bias': model.conv3.bias.cpu().clone(),
        'fc1.weight': model.fc1.weight.cpu().clone(),
        'fc1.bias': model.fc1.bias.cpu().clone(),
        'fc2.weight': model.fc2.weight.cpu().clone(),
        'fc2.bias': model.fc2.bias.cpu().clone(),
    }
    torch.save(model_state, model_file)
    print(f"模型已保存到: {model_file}")


if __name__ == '__main__':
    main()
