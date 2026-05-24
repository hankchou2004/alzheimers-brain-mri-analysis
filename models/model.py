from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models



class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate, kernel, pooling, BN=True, relu_type='leaky'):
        super().__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        pool_kernel, pool_stride, pool_padding = pooling
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, kernel_stride, kernel_padding, bias=False)
        self.BN = nn.BatchNorm3d(out_channels) # 保持存在
        self.relu = nn.LeakyReLU() if relu_type=='leaky' else nn.ReLU()
        self.pooling = nn.MaxPool3d(pool_kernel, pool_stride, pool_padding)
        self.dropout = nn.Dropout(drop_rate) 
       
    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)      # 1. 先做標準化，穩定分佈
        x = self.relu(x)    # 2. 再做激活，保留非線性
        x = self.pooling(x) # 3. 最後做池化，減少維度
        x = self.dropout(x)
        return x



class CNN(nn.Module):
    def __init__(self, fil_num, drop_rate):
        super(CNN, self).__init__()
        self.block1 = ConvLayer(1, fil_num, 0.1, (7, 2, 0), (3, 2, 0))
        self.block2 = ConvLayer(fil_num, 2*fil_num, 0.1, (4, 1, 0), (2, 2, 0))
        self.block3 = ConvLayer(2*fil_num, 4*fil_num, 0.1, (3, 1, 0), (2, 2, 0))
        self.block4 = ConvLayer(4*fil_num, 8*fil_num, 0.1, (3, 1, 0), (2, 1, 0))
        self.dense1 = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(8*fil_num*6*8*6, 30),  # 🔑 改回 46080，這樣才能載入權重
        )
        self.dense2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(30, 2),
        )

    def forward(self, x, return_features=False, return_adapted_features=False):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        features = self.block4(x)

        adapted_features = F.interpolate(features, size=(6,8,6), mode='trilinear', align_corners=False)
        batch_size = adapted_features.shape[0]
        flat_features = adapted_features.view(batch_size, -1)

        x = self.dense1(flat_features)
        logits = self.dense2(x)

        if return_adapted_features:
            return logits, adapted_features
        if return_features:
            return logits, features
        return logits



import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class ResNet18_3D(nn.Module):
    def __init__(self, fil_num, drop_rate):
        super(ResNet18_3D, self).__init__()
        
        # 載入官方預訓練的 R3D_18 (18層 3D ResNet)
        # weights=R3D_18_Weights.DEFAULT 會下載 ImageNet/Kinetics 預訓練權重
        self.model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        
        # 1. 修改輸入層 (Stem): 原始為 3 channel (RGB)，改為 input_channels (如: 1)
        # r3d_18 的結構中，第一層在 model.stem[0]
        self.model.stem[0] = nn.Conv3d(
            1, 64, 
            kernel_size=(3, 7, 7), 
            stride=(1, 2, 2), 
            padding=(1, 3, 3), 
            bias=False
        )
        
        # 2. 修改分類層 (Heads): 將最後的 fc 改為輸出 num_classes (如: 2)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(num_ftrs, 2)
        )
    
    def forward(self, x, return_features=False, target_layer_name='layer4'):
        # 1. 進入 Stem (包含你修改過的 Conv3d)
        x = self.model.stem(x)
        
        # 2. 逐層提取特徵，並存入字典
        l1 = self.model.layer1(x)
        l2 = self.model.layer2(l1)
        l3 = self.model.layer3(l2)
        l4 = self.model.layer4(l3)

        # 🔑 建立對照表，對應你想要查看的 "Decoder" 層級
        feature_map_dict = {
            'layer2': l2, 
            'layer3': l3, 
            'layer4': l4
        }
        
        # 取得目標特徵圖
        target_features = feature_map_dict.get(target_layer_name, l4)
        
        # 3. 分類頭 (使用最後一層 l4 進行預測)
        x = self.model.avgpool(l4)
        x = torch.flatten(x, 1)
        logits = self.model.fc(x)
        
        if return_features:
            # 回傳預測結果與指定的特徵圖
            return logits, target_features
        
        return logits
    
