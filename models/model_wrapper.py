import os
import csv
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
# 假設這些工具函數都在 utils.py 中
from utils.utils import (
    matrix_sum, get_accu, 
    get_confusion_matrix
)
from data.dataloader import Data

import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 計算 Cross Entropy Loss (不進行 reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 取得 p_t (正確類別的預測機率)
        pt = torch.exp(-ce_loss)
        
        # 計算 Focal Loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ModelWrapper:
    def __init__(self,
                 fil_num,
                 drop_rate,
                 seed,
                 batch_size,
                 balanced,
                 Data_dir,
                 exp_idx,
                 model,
                 metric,
                 pretrained_path=None
                 ):

        self.seed = seed
        self.exp_idx = exp_idx
        
        
        
        # 🔑 自動取得模型類別名稱 (例如 "DualStreamCNN")
        if hasattr(model, '__name__'):
            self.model_name = model.__name__      # 如果傳入的是類別 (例如 CNN)
        else:
            self.model_name = model.__class__.__name__ # 如果傳入的是實例

        # 根據 metric 設定評估函數
        self.eval_metric = get_accu 

        self.model = model(fil_num=fil_num, drop_rate=drop_rate).cuda()

       # 🔑 2. 修正後：載入預訓練權重並過濾分類層
        if pretrained_path:
            if os.path.exists(pretrained_path):
                print(f"🔄 正在從 {pretrained_path} 進行遷移學習載入...")
                checkpoint = torch.load(pretrained_path, map_location='cuda')
                
                # 取得 state_dict
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    pretrained_dict = checkpoint['state_dict']
                else:
                    pretrained_dict = checkpoint
                
                # 獲取目前模型的 state_dict
                model_dict = self.model.state_dict()

                # --- 關鍵動作：過濾掉分類層 ---
                # 假設你的分類層名稱包含 'dense2' (請根據你的 model.py 修改關鍵字)
                filtered_dict = {
                    k: v for k, v in pretrained_dict.items() 
                    if k in model_dict and 'dense2' not in k
                }

                # 更新現有的 model_dict
                model_dict.update(filtered_dict)
                
                # 載入更新後的權重 (strict=False 允許部分 key 不匹配)
                self.model.load_state_dict(model_dict, strict=False)
                
                print(f"✅ 成功載入 {len(filtered_dict)} 個層的權重，分類層已跳過並重新初始化。")
            else:
                print(f"⚠️ 警告：找不到路徑 {pretrained_path}，將使用隨機初始化。")

        # 初始化 Data (這部分沿用你原本的邏輯)
        self.prepare_dataloader(batch_size, balanced, Data_dir)
        
        #  補齊：將 Data 路徑存入 self 供 evaluate_and_save 使用
        self.data_dir = Data_dir
        
        # 根據動態名稱設定路徑
        self.checkpoint_dir = f'./checkpoint_dir/{self.model_name}_exp{exp_idx}'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_metric_path = None
        self.best_loss_path = None
        # 修改後：加上模型名稱與實驗參數
        self.log_path = os.path.join(self.checkpoint_dir, f'{self.model_name}_training_log.csv')
        self._init_log()
        
    # gradcam_triptych.py 或 wrapper 所在檔案
    def _init_log(self):
        with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 🔑 確保這裡的名稱與 plot.py 一致
            writer.writerow([
                'epoch', 'train_loss', 'train_acc', 
                'valid_loss', 'valid_acc', 'valid_matrix', 'lr'
            ])

    # 修正後的指標計算 (增加 epsilon 防止除以 0)
    def get_precision_recall_f1(self, matrix):
        eps = 1e-9

        # 根據 matrix[Predict][Actual] 的邏輯明確定義：
        tn = matrix[0][0] # 預測 0，實際 0
        fn = matrix[0][1] # 預測 0，實際 1 (漏看)
        fp = matrix[1][0] # 預測 1，實際 0 (誤報)
        tp = matrix[1][1] # 預測 1，實際 1

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        specificity = tn / (tn + fp + eps)

        return precision, recall, f1, specificity

    def prepare_dataloader(self, batch_size, balanced, data_dir):
        train_data = Data(data_dir, self.exp_idx, stage='train', seed=self.seed,apply_aug=True)
        valid_data = Data(data_dir, self.exp_idx, stage='valid', seed=self.seed)
        test_data  = Data(data_dir, self.exp_idx, stage='test', seed=self.seed)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()

        # 策略選擇 (簡化邏輯)
        if balanced == 0: # 不平衡
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        elif balanced == 1: # Weighted Sampler
            sampler = torch.utils.data.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            self.imbalanced_ratio = 1.0
        
        self.valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    def train(self, lr, epochs):
        torch.cuda.empty_cache()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, self.imbalanced_ratio])).cuda()

        # 🔑 改用 Focal Loss
        # alpha 可以設定為類別權重，例如你原本的 imbalanced_ratio
        # 如果你的 imbalanced_ratio = 5 (正樣本很少)，可以設 alpha=imbalanced_ratio
        #self.criterion = FocalLoss(alpha=self.imbalanced_ratio, gamma=2).cuda()

        # 🔑 實作 ReduceLROnPlateau
        # mode='min': 因為我們監控的是 Loss，越小越好
        # factor=0.5: 觸發時學習率減半 (相當於 gamma=0.5)
        # patience=5: 如果連續 5 個 Epoch 的 Valid Loss 都沒有下降，就觸發降速
        # min_lr=1e-6: 學習率降到的最低底限
        # verbose=True: 在舊版 PyTorch 會印出訊息，新版建議手動 print
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-8
        )

        # 🔑 1. 初始化 AMP 的 GradScaler
        self.scaler = torch.amp.GradScaler('cuda')

        self.optimal_valid_metric = 0
        self.best_loss = float('inf')

        for self.epoch in range(epochs):
            train_loss, train_acc = self._run_epoch(stage='train')
            valid_loss, valid_matrix, valid_acc = self._run_epoch(stage='valid')

            # 🔑 執行學習率更新
            # 🔑 重要：ReduceLROnPlateau 的 step 需要傳入監控的數值
            # 這裡我們傳入剛剛計算出的 valid_loss
            self.scheduler.step(valid_loss)

            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"Epoch {self.epoch} | T_Loss: {train_loss:.4f} | V_Loss: {valid_loss:.4f} | V_Acc: {valid_acc:.4f}")

            # 記錄日誌
            with open(self.log_path, 'a', newline='') as f:
                csv.writer(f).writerow([self.epoch, train_loss, train_acc, valid_loss, valid_acc, valid_matrix,current_lr])

            # 儲存最佳模型 (Metric)
            curr_metric = self.eval_metric(valid_matrix)
            if curr_metric > self.optimal_valid_metric:
                self.optimal_valid_metric = curr_metric
                self.best_metric_path = self._save_model('best_metric')

            # 儲存最佳模型 (Loss)
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.best_loss_path = self._save_model('best_loss')

    def _run_epoch(self, stage):
        """整合 Train 與 Valid 的邏輯"""
        is_train = (stage == 'train')
        self.model.train(is_train)
        loader = self.train_dataloader if is_train else self.valid_dataloader
        
        running_loss, correct, total = 0.0, 0, 0
        matrix = [[0, 0], [0, 0]]

        with torch.set_grad_enabled(is_train):
            for inputs, labels in tqdm(loader, desc=f"{stage.capitalize()} Epoch {self.epoch}"):
                inputs, labels = inputs.cuda(), labels.cuda()
                
                if is_train: self.optimizer.zero_grad()

                # 🔑 2. 使用 autocast 進行混合精度正向傳播
                with torch.amp.autocast('cuda'):
                    preds = self.model(inputs)
                    loss = self.criterion(preds, labels)
                
               
                
                if is_train:
                    # 🔑 3. 使用 scaler 縮放 loss 並進行反向傳播
                    self.scaler.scale(loss).backward()
                    
                    # 🔑 4. scaler.step() 會先縮放回原梯度，若梯度沒爆炸則執行 optimizer.step()
                    self.scaler.step(self.optimizer)
                    
                    # 🔑 5. 更新 scaler 因子
                    self.scaler.update()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(preds, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                if not is_train:
                    matrix = matrix_sum(matrix, get_confusion_matrix(preds, labels))

        if is_train:
            return running_loss / total, correct / total
        else:
            return running_loss / total, matrix, correct / total

    def _save_model(self, tag):
        path = os.path.join(self.checkpoint_dir, f'{self.model_name}_{tag}.pth')
        # 移除舊檔案以節省空間 (可選)
        for old_file in glob.glob(os.path.join(self.checkpoint_dir, f'{self.model_name}_{tag}_*.pth')):
            os.remove(old_file)
        
        new_path = path.replace('.pth', f'_{self.epoch}.pth')
        torch.save(self.model.state_dict(), new_path)
        return new_path

    def test(self):
        """測試最佳模型並產出 CSV"""
        # 遍歷儲存的最佳權重路徑
        for tag, path in [('best_metric', self.best_metric_path), ('best_loss', self.best_loss_path)]:
            if path: 
                print(f"🧐 正在評估 {tag} 模型: {os.path.basename(path)}")
                self.evaluate_and_save(path, tag)

    def evaluate_and_save(self, model_path, tag):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # 2. 🔑 修正：CSV 檔名加上模型名稱
        csv_name = f'{self.model_name}_test_{tag}.csv'
        csv_path = os.path.join(self.checkpoint_dir, csv_name)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # 🔑 確保這裡也有 'sensitivity' 欄位
            writer.writerow(['model_type', 'stage', 'acc', 'precision', 'recall', 'specificity', 'f1', 'confusion_matrix'])

            for stage in ['train', 'valid', 'test']:
                data = Data(self.data_dir, self.exp_idx, stage=stage, seed=self.seed)
                loader = DataLoader(data, batch_size=1, shuffle=False)
                
                matrix = [[0, 0], [0, 0]]
                with torch.no_grad():
                    for inputs, labels in loader:
                        inputs, labels = inputs.cuda(), labels.cuda()
                        preds = self.model(inputs)
                        matrix = matrix_sum(matrix, get_confusion_matrix(preds, labels))
                
                # 4. 取得各項醫學指標
                acc = get_accu(matrix)
                # 假設此函數回傳 prec, sens, f1, spec
                prec, sens, f1, spec = self.get_precision_recall_f1(matrix) 
                
                # 5. 🔑 修正：確保寫入的數值順序與 Header 完全對應 (共 9 個值)
                writer.writerow([
                    tag,           # model_type
                    stage,         # stage
                    acc,           # acc
                    prec,          # precision
                    sens,          # recall
                    spec,          # specificity
                    f1,            # f1
                    str(matrix)    # confusion_matrix
                ])
        
        print(f"✅ 測試結果已儲存至: {csv_path}")