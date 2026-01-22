import os
import argparse
import json
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================
# TODO: 请修改为你的队长学号
# ==========================================
STUDENT_ID = "PB23000073"  # <--- 修改这里
MODEL_FILE = "model_weights.pth"

# ==========================================
# 1. 配置与参数解析
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test dataset root (containing img folder)')
args = parser.parse_args()

CONFIG = {
    "img_size": 128,       # 必须与训练时保持一致
    "batch_size": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==========================================
# 2. 推理专用数据集 (不读取 Label)
# ==========================================
class InferenceDataset(Dataset):
    def __init__(self, root_dir):
        """
        根据要求，测试集只包含 img 子目录，不包含 txt 目录。
        因此这里只读取图片，不查找标签。
        """
        self.img_dir = os.path.join(root_dir, 'img')
        
        if not os.path.exists(self.img_dir):
            raise ValueError(f"Image directory not found: {self.img_dir}")
            
        self.img_paths = glob.glob(os.path.join(self.img_dir, '*.png'))
        self.img_paths.sort() # 排序以保证顺序确定性

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # 获取文件名作为 Key (不带后缀)
        basename = os.path.basename(img_path)
        file_id = os.path.splitext(basename)[0]
        
        # 图片预处理 (必须与训练时完全一致)
        img = cv2.imread(img_path)
        if img is None:
            # 简单的错误处理，防止坏图导致崩溃
            img = np.zeros((CONFIG['img_size'], CONFIG['img_size'], 3), dtype=np.uint8)
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (CONFIG['img_size'], CONFIG['img_size']))
        
        # 归一化 [0, 1] 并转为 Tensor (C, H, W)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        return img, file_id

# ==========================================
# 3. 网络结构定义 (必须与 main.py 一致)
# ==========================================

class Layer:
    def forward(self, x): pass
    # 推理阶段不需要 backward 和 step

class Linear(Layer):
    def __init__(self, in_features, out_features):
        # 初始化占位，具体数值会被 load_weights 覆盖
        self.W = torch.zeros(in_features, out_features).to(CONFIG['device'])
        self.b = torch.zeros(1, out_features).to(CONFIG['device'])

    def forward(self, x):
        return torch.matmul(x, self.W) + self.b

class ReLU(Layer):
    def forward(self, x):
        return torch.clamp(x, min=0)

class Sigmoid(Layer):
    def forward(self, x):
        return 1 / (1 + torch.exp(-x))

class Flatten(Layer):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class Conv2d_Manual(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 初始化占位
        self.W = torch.zeros(out_channels, in_channels, kernel_size, kernel_size).to(CONFIG['device'])
        self.b = torch.zeros(out_channels).to(CONFIG['device'])

    def forward(self, x):
        # Im2Col / Unfold 实现
        x_unfolded = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        w_flat = self.W.view(self.out_channels, -1)
        # Matrix Mult
        out_unfolded = torch.matmul(w_flat, x_unfolded) + self.b.view(1, -1, 1)
        # Fold / Reshape
        h_out = (x.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (x.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = out_unfolded.view(x.shape[0], self.out_channels, h_out, w_out)
        return output

class MaxPool2d(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # 推理时不需要 indices
        return F.max_pool2d(x, self.kernel_size, self.stride)

class CNNModel:
    def __init__(self):
        # 结构必须与训练时完全一致
        self.layers = [
            Conv2d_Manual(3, 8, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(2, 2), # 128->64
            
            Conv2d_Manual(8, 16, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(2, 2), # 64->32
            
            Conv2d_Manual(16, 32, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(2, 2), # 32->16
            
            Flatten(),
            Linear(32 * 16 * 16, 64),
            ReLU(),
            Linear(64, 1),
            Sigmoid()
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def load_weights(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model weights file not found: {path}")
        
        # 加载 list of dicts
        saved_layers = torch.load(path, map_location=CONFIG['device'])
        
        loaded_count = 0
        for layer, params in zip(self.layers, saved_layers):
            if 'W' in params and hasattr(layer, 'W'):
                layer.W = params['W'].to(CONFIG['device'])
                layer.b = params['b'].to(CONFIG['device'])
                loaded_count += 1
        print(f"Successfully loaded weights for {loaded_count} layers.")

# ==========================================
# 4. 主程序：推理并生成 JSON
# ==========================================

def run_inference():
    # 1. 准备数据
    print(f"Reading test data from: {args.test_data_path}")
    dataset = InferenceDataset(args.test_data_path)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 2. 准备模型
    print("Initializing model...")
    model = CNNModel()
    
    # 加载权重
    print(f"Loading weights from: {MODEL_FILE}")
    try:
        model.load_weights(MODEL_FILE)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'model_weights.pth' exists and matches the architecture.")
        return

    # 3. 推理循环
    results = {}
    print("Starting inference...")
    
    # 不需要计算梯度，节省显存
    with torch.no_grad():
        for imgs, file_ids in tqdm(dataloader):
            imgs = imgs.to(CONFIG['device'])
            
            # Forward pass
            outputs = model.forward(imgs)
            
            # 这里的 outputs 是 sigmoid 后的概率 (0~1)
            # 大于 0.5 判定为 Defective (True)
            predictions = (outputs > 0.5).cpu().numpy().flatten()
            
            # 存入字典
            for fid, is_defective in zip(file_ids, predictions):
                results[fid] = bool(is_defective == 1.0)

    # 4. 保存结果
    output_filename = f"{STUDENT_ID}.json"
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nInference completed.")
    print(f"Results saved to: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    run_inference()