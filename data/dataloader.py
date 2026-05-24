import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from utils.utils import read_csv # 確保 utils 有這個 function

class MRI_Transform:
    """專為 3D MRI 設計的數據增強類別"""
    def __init__(self, flip_prob=0.5, noise_std=0.01, brightness_range=0.2):
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.brightness_range = brightness_range

    def __call__(self, x):
        # x shape: (1, D, H, W)
        
        # 1. 隨機左右翻轉 (假設 W 軸是左右)
        if random.random() < self.flip_prob:
            x = torch.flip(x, dims=[3])
            
        # 2. 隨機上下翻轉 (假設 H 軸是上下)
        if random.random() < self.flip_prob:
            x = torch.flip(x, dims=[2])
            
        # 3. 隨機加高斯噪聲
        if random.random() < self.flip_prob:
            x = x + torch.randn_like(x) * self.noise_std
            
        # 4. 隨機亮度調整
        if random.random() < self.flip_prob:
            factor = 1.0 + (random.random() - 0.5) * self.brightness_range
            x = x * factor
            
        return x

class Data(Dataset):
    def __init__(self, data_dir, exp_idx, stage, seed=1000, apply_aug=True):
        random.seed(seed)
        torch.manual_seed(seed)
        
        self.data_dir = data_dir
        self.stage = stage
        self.apply_aug = apply_aug
        
        # 使用 os.path.join 組合路徑
        csv_path = os.path.join('./lookupcsv', f'exp{exp_idx}', f'{stage}.csv')
        self.data_list, self.label_list = read_csv(csv_path)

        # 宣告增強工具
        self.transform = MRI_Transform()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        label = self.label_list[idx]
        
        # 讀取數據並轉為 Tensor
        npy_path = os.path.join(self.data_dir, f"{self.data_list[idx]}.npy")
        try:
            data = np.load(npy_path).astype(np.float32)
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing file: {npy_path}")

        # 擴展維度為 (Channel=1, D, H, W)
        data = torch.from_numpy(data).unsqueeze(0)

        # 僅在訓練階段且針對特定類別 (如 AD, label=1) 進行增強
        # 如果你想對所有訓練資料增強，可以移除 label == 1 的判斷
        if self.stage == 'train' and self.apply_aug:
            if label == 1 : 
                data = self.transform(data)

        return data, label

    def get_sample_weights(self):
        """計算用於 WeightedRandomSampler 的權重與類別比例"""
        labels = np.array(self.label_list)
        count0 = np.sum(labels == 0)
        count1 = np.sum(labels == 1)
        total = len(labels)
        
        # 避免除以零
        w0 = total / count0 if count0 > 0 else 1.0
        w1 = total / count1 if count1 > 0 else 1.0
        
        weights = [w0 if l == 0 else w1 for l in labels]
        imbalanced_ratio = count0 / count1 if count1 > 0 else 1.0
        
        return weights, imbalanced_ratio