"""
CNN模型实现 - 手动实现前向和反向传播
不使用autograd，完全手动计算梯度
"""
import torch
import numpy as np
import math


class Conv2d:
    """手动实现的2D卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # 初始化权重和偏置
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        limit = math.sqrt(6.0 / fan_in)
        self.weight = torch.randn(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]) * limit
        self.bias = torch.zeros(out_channels)
        
        # 梯度
        self.weight_grad = None
        self.bias_grad = None
        self.input_cache = None
    
    def forward(self, x):
        """
        前向传播
        x: (batch, in_channels, H, W)
        返回: (batch, out_channels, H_out, W_out)
        """
        self.input_cache = x.clone()
        batch_size, in_ch, in_h, in_w = x.shape
        
        # 计算输出尺寸
        out_h = (in_h + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (in_w + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        # 添加padding
        if self.padding[0] > 0 or self.padding[1] > 0:
            x_padded = torch.zeros(batch_size, in_ch, in_h + 2*self.padding[0], in_w + 2*self.padding[1], device=x.device, dtype=x.dtype)
            x_padded[:, :, self.padding[0]:in_h+self.padding[0], self.padding[1]:in_w+self.padding[1]] = x
            x = x_padded
        
        # 执行卷积
        output = torch.zeros(batch_size, self.out_channels, out_h, out_w, device=x.device, dtype=x.dtype)
        
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * self.stride[0]
                        h_end = h_start + self.kernel_size[0]
                        w_start = ow * self.stride[1]
                        w_end = w_start + self.kernel_size[1]
                        
                        x_patch = x[b, :, h_start:h_end, w_start:w_end]
                        output[b, oc, oh, ow] = torch.sum(x_patch * self.weight[oc]) + self.bias[oc]
        
        return output
    
    def backward(self, grad_output):
        """
        反向传播
        grad_output: (batch, out_channels, H_out, W_out)
        返回: grad_input (batch, in_channels, H, W)
        """
        batch_size, out_ch, out_h, out_w = grad_output.shape
        x = self.input_cache
        batch_size_in, in_ch, in_h, in_w = x.shape
        
        # 初始化梯度
        grad_input = torch.zeros_like(x)
        self.weight_grad = torch.zeros_like(self.weight)
        self.bias_grad = torch.zeros_like(self.bias)
        
        # 处理padding - 重新计算前向传播时的输入（带padding）
        if self.padding[0] > 0 or self.padding[1] > 0:
            x_padded = torch.zeros(batch_size_in, in_ch, in_h + 2*self.padding[0], in_w + 2*self.padding[1], device=x.device, dtype=x.dtype)
            x_padded[:, :, self.padding[0]:in_h+self.padding[0], self.padding[1]:in_w+self.padding[1]] = x
            x_forward = x_padded
            in_h_padded = in_h + 2 * self.padding[0]
            in_w_padded = in_w + 2 * self.padding[1]
        else:
            x_forward = x
            in_h_padded = in_h
            in_w_padded = in_w
        
        # 创建带padding的grad_input用于计算
        if self.padding[0] > 0 or self.padding[1] > 0:
            grad_input_padded = torch.zeros(batch_size_in, in_ch, in_h_padded, in_w_padded, device=x.device, dtype=x.dtype)
        else:
            grad_input_padded = grad_input
        
        # 计算权重和偏置的梯度
        for b in range(batch_size):
            for oc in range(out_ch):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * self.stride[0]
                        h_end = h_start + self.kernel_size[0]
                        w_start = ow * self.stride[1]
                        w_end = w_start + self.kernel_size[1]
                        
                        grad = grad_output[b, oc, oh, ow]
                        
                        # 偏置梯度
                        self.bias_grad[oc] += grad
                        
                        # 权重梯度
                        if h_end <= in_h_padded and w_end <= in_w_padded:
                            x_patch = x_forward[b, :, h_start:h_end, w_start:w_end]
                            self.weight_grad[oc] += grad * x_patch
                        
                        # 输入梯度
                        if h_end <= in_h_padded and w_end <= in_w_padded:
                            weight_patch = self.weight[oc]
                            grad_input_padded[b, :, h_start:h_end, w_start:w_end] += grad * weight_patch
        
        # 如果使用了padding，需要裁剪
        if self.padding[0] > 0 or self.padding[1] > 0:
            grad_input = grad_input_padded[:, :, self.padding[0]:in_h+self.padding[0], self.padding[1]:in_w+self.padding[1]]
        
        return grad_input


class MaxPool2d:
    """手动实现的2D最大池化层"""
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is None else (stride if isinstance(stride, tuple) else (stride, stride))
        if self.stride is None:
            self.stride = self.kernel_size
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.input_cache = None
        self.max_indices_cache = None
    
    def forward(self, x):
        """前向传播"""
        self.input_cache = x.clone()
        batch_size, channels, in_h, in_w = x.shape
        
        # 添加padding
        if self.padding[0] > 0 or self.padding[1] > 0:
            x_padded = torch.zeros(batch_size, channels, in_h + 2*self.padding[0], in_w + 2*self.padding[1], device=x.device, dtype=x.dtype)
            x_padded[:, :, self.padding[0]:in_h+self.padding[0], self.padding[1]:in_w+self.padding[1]] = x
            x = x_padded
            in_h = in_h + 2 * self.padding[0]
            in_w = in_w + 2 * self.padding[1]
        
        # 计算输出尺寸
        out_h = (in_h - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (in_w - self.kernel_size[1]) // self.stride[1] + 1
        
        output = torch.zeros(batch_size, channels, out_h, out_w, device=x.device, dtype=x.dtype)
        self.max_indices_cache = {}
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * self.stride[0]
                        h_end = h_start + self.kernel_size[0]
                        w_start = ow * self.stride[1]
                        w_end = w_start + self.kernel_size[1]
                        
                        patch = x[b, c, h_start:h_end, w_start:w_end]
                        max_val, max_idx = torch.max(patch.view(-1), dim=0)
                        output[b, c, oh, ow] = max_val
                        
                        # 保存最大值位置
                        max_idx_2d = max_idx.item()
                        max_h = max_idx_2d // self.kernel_size[1]
                        max_w = max_idx_2d % self.kernel_size[1]
                        self.max_indices_cache[(b, c, oh, ow)] = (h_start + max_h, w_start + max_w)
        
        return output
    
    def backward(self, grad_output):
        """反向传播"""
        batch_size, channels, out_h, out_w = grad_output.shape
        x = self.input_cache
        batch_size_in, channels_in, in_h, in_w = x.shape
        
        grad_input = torch.zeros_like(x)
        
        # 处理padding
        if self.padding[0] > 0 or self.padding[1] > 0:
            grad_input_padded = torch.zeros(batch_size_in, channels_in, in_h + 2*self.padding[0], in_w + 2*self.padding[1], device=x.device, dtype=x.dtype)
            in_h_orig = in_h
            in_w_orig = in_w
            in_h = in_h + 2 * self.padding[0]
            in_w = in_w + 2 * self.padding[1]
        else:
            grad_input_padded = grad_input
            in_h_orig = in_h
            in_w_orig = in_w
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        if (b, c, oh, ow) in self.max_indices_cache:
                            max_h, max_w = self.max_indices_cache[(b, c, oh, ow)]
                            grad_input_padded[b, c, max_h, max_w] += grad_output[b, c, oh, ow]
        
        # 如果使用了padding，需要裁剪
        if self.padding[0] > 0 or self.padding[1] > 0:
            grad_input = grad_input_padded[:, :, self.padding[0]:in_h_orig+self.padding[0], self.padding[1]:in_w_orig+self.padding[1]]
        else:
            grad_input = grad_input_padded
        
        return grad_input


class ReLU:
    """ReLU激活函数"""
    def __init__(self):
        self.input_cache = None
    
    def forward(self, x):
        self.input_cache = x.clone()
        return torch.where(x > 0, x, torch.zeros_like(x))
    
    def backward(self, grad_output):
        return grad_output * torch.where(self.input_cache > 0, torch.ones_like(self.input_cache), torch.zeros_like(self.input_cache))


class Linear:
    """全连接层"""
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        # 初始化权重
        limit = math.sqrt(6.0 / (in_features + out_features))
        self.weight = torch.randn(out_features, in_features) * limit
        self.bias = torch.zeros(out_features)
        
        self.weight_grad = None
        self.bias_grad = None
        self.input_cache = None
    
    def forward(self, x):
        """前向传播: x (batch, in_features) -> (batch, out_features)"""
        self.input_cache = x.clone()
        return torch.matmul(x, self.weight.t()) + self.bias
    
    def backward(self, grad_output):
        """
        反向传播
        grad_output: (batch, out_features)
        返回: (batch, in_features)
        """
        batch_size = grad_output.shape[0]
        
        # 计算梯度
        self.weight_grad = torch.matmul(grad_output.t(), self.input_cache)
        self.bias_grad = torch.sum(grad_output, dim=0)
        grad_input = torch.matmul(grad_output, self.weight)
        
        return grad_input


class Flatten:
    """展平层"""
    def __init__(self):
        self.input_shape = None
    
    def forward(self, x):
        self.input_shape = x.shape
        return x.view(x.shape[0], -1)
    
    def backward(self, grad_output):
        return grad_output.view(self.input_shape)


class CNN:
    """完整的CNN模型"""
    def __init__(self):
        # 卷积层1: 320x320 -> 160x160 (padding=1, stride=2)
        self.conv1 = Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)  # 160x160 -> 80x80
        
        # 卷积层2: 80x80 -> 40x40
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)  # 40x40 -> 20x20
        
        # 卷积层3: 20x20 -> 10x10
        self.conv3 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu3 = ReLU()
        self.pool3 = MaxPool2d(kernel_size=2, stride=2)  # 10x10 -> 5x5
        
        # 全连接层
        # 经过池化后: 5x5x128 = 3200
        self.flatten = Flatten()
        self.fc1 = Linear(3200, 512)
        self.relu4 = ReLU()
        self.fc2 = Linear(512, 1)
        
        self.training = True
    
    def forward(self, x):
        """前向传播"""
        # x: (batch, 3, 320, 320)
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)  # (batch, 32, 79, 79)
        
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)  # (batch, 64, 20, 20)
        
        x = self.conv3.forward(x)
        x = self.relu3.forward(x)
        x = self.pool3.forward(x)  # (batch, 128, 5, 5)
        
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu4.forward(x)
        x = self.fc2.forward(x)
        
        return x
    
    def backward(self, grad_output):
        """反向传播"""
        grad = self.fc2.backward(grad_output)
        grad = self.relu4.backward(grad)
        grad = self.fc1.backward(grad)
        grad = self.flatten.backward(grad)
        
        grad = self.pool3.backward(grad)
        grad = self.relu3.backward(grad)
        grad = self.conv3.backward(grad)
        
        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)
        
        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)
        
        return grad
    
    def get_parameters(self):
        """获取所有可训练参数"""
        params = []
        params.append(('conv1.weight', self.conv1.weight))
        params.append(('conv1.bias', self.conv1.bias))
        params.append(('conv2.weight', self.conv2.weight))
        params.append(('conv2.bias', self.conv2.bias))
        params.append(('conv3.weight', self.conv3.weight))
        params.append(('conv3.bias', self.conv3.bias))
        params.append(('fc1.weight', self.fc1.weight))
        params.append(('fc1.bias', self.fc1.bias))
        params.append(('fc2.weight', self.fc2.weight))
        params.append(('fc2.bias', self.fc2.bias))
        return params
    
    def zero_grad(self):
        """清零梯度"""
        self.conv1.weight_grad = None
        self.conv1.bias_grad = None
        self.conv2.weight_grad = None
        self.conv2.bias_grad = None
        self.conv3.weight_grad = None
        self.conv3.bias_grad = None
        self.fc1.weight_grad = None
        self.fc1.bias_grad = None
        self.fc2.weight_grad = None
        self.fc2.bias_grad = None
    
    def train(self):
        """设置为训练模式"""
        self.training = True
    
    def eval(self):
        """设置为评估模式"""
        self.training = False
