import os
import csv
import json
import time
import torch
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold

# --- 1. 檔案讀寫與配置 ---

def read_json(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

import csv

import csv
import os

def read_csv(filename, task_type='AD/CN'):
    """
    統一的 CSV 讀取函數
    支援二分類、一對多 (OvR) 以及 三分類任務
    """
    # 擴充後的任務設定
    task_config = {
        # --- 原有二分類任務 ---
        'AD/CN':   {'AD': 1, 'NL': 0, 'CN': 0},
        'MCI/CN':  {'MCI': 1, 'NL': 0, 'CN': 0},
        'AD/MCI':  {'AD': 1, 'MCI': 0},
        
    }
    
    if task_type not in task_config:
        raise ValueError(f"不支援的任務類型: {task_type}")
        
    label_map = task_config[task_type]
    target_categories = set(label_map.keys())
    
    filenames = []
    labels = []

    if not os.path.exists(filename):
        print(f"❌ 錯誤：找不到檔案 {filename}")
        return [], []

    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader) 
        except StopIteration:
            return [], []
        
        for row in reader:
            if not row or len(row) < 3: continue
            
            img_id = row[0]
            diag = row[2] 
            
            # 檢查該列的診斷是否在我們定義的任務中
            if diag in target_categories:
                filenames.append(img_id) 
                labels.append(label_map[diag])
                
    print(f"\n✅ 任務 [{task_type}] 讀取完成：")
    
    # 動態統計所有出現過的標籤
    unique_labels = sorted(list(set(label_map.values())))
    for val in unique_labels:
        count = labels.count(val)
        # 找出哪些類別名稱被歸類到這個標籤數值
        cats = [k for k, v in label_map.items() if v == val]
        print(f"   - Label {val} ({'/'.join(cats)}): {count} 筆")
        
    return filenames, labels
# --- 2. 矩陣與指標計算 (優化效能) ---

def get_confusion_matrix(preds, labels):
    """使用 Numpy 加速計算混淆矩陣"""
    preds_idx = torch.argmax(preds, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    
    matrix = np.zeros((2, 2), dtype=int)
    for p, l in zip(preds_idx, labels):
        matrix[p, l] += 1  # matrix[预测][真实]
    return matrix

def matrix_sum(A, B):
    return (np.array(A) + np.array(B)).tolist()

def get_accu(matrix):
    m = np.array(matrix)
    return float(np.sum(np.diag(m))) / (np.sum(m) + 1e-9)

# --- 3. 資料切分 (支援 K-Fold) ---

import os
import csv
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold

def data_split(repe_time, n_splits=5, test_num=0):
    input_csv = os.path.join('lookupcsv', 'ADNI.csv')
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found!")
        return

    with open(input_csv, 'r', encoding='utf-8') as f:
        all_rows = list(csv.reader(f))
    
    header = all_rows[0]
    # 注意：根據你的截圖，Group 在第 3 欄，索引為 2
    label_map = {'NL': 0, 'CN': 0, 'MCI': 1, 'AD': 2}

    # 1. 先把所有有正確標籤的資料過濾出來 (從 index 1 開始跳過 header)
    # 這裡將 row[1] 改為 row[2]
    valid_rows = [row for row in all_rows[1:] if row[2] in label_map]

    if len(valid_rows) == 0:
        print("❌ 錯誤：找不到任何符合標籤 (NL, CN, MCI, AD) 的資料，請檢查 CSV 欄位索引！")
        return

    # 2. 根據 test_num 切分開發集與測試集
    # 如果 test_num=674，則前 674 筆為開發集，剩下為測試集
    # (或者依你需求調整：如果是要後 674 筆當測試，就把切割點往前移)
    dev_data = valid_rows[:test_num]
    test_data = valid_rows[test_num:]

    if len(dev_data) == 0:
        print(f"❌ 錯誤：開發集 (dev_data) 為空！請檢查 test_num ({test_num}) 設定是否正確。")
        print(f"目前總有效資料筆數：{len(valid_rows)}")
        return

    X_dev = np.array(dev_data)
    # 這裡也要同步改為 row[2]
    y_dev = np.array([label_map[row[2]] for row in dev_data])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(X_dev, y_dev))

    print("\n" + "="*50)
    print(f"📊 資料切割總覽 (3-Class)")
    print(f"開發集 (Dev/Valid): {len(dev_data)} 筆")
    print(f"固定測試集 (Test): {len(test_data)} 筆")
    print("="*50)

    # --- 內部統計函數也要修 ---
    def count_classes(data_list):
        # 這裡也要改 row[2]
        labels = [label_map[r[2]] for r in data_list]
        c = Counter(labels)
        return [c.get(0, 0), c.get(1, 0), c.get(2, 0)]

    for i in range(repe_time):
        fold_idx = i % n_splits
        train_idx, valid_idx = folds[fold_idx]
        
        train_set = X_dev[train_idx].tolist()
        valid_set = X_dev[valid_idx].tolist()

        train_counts = count_classes(train_set)
        valid_counts = count_classes(valid_set)
        test_counts  = count_classes(test_data)

        # 寫入檔案 (後續邏輯不變...)
        folder = os.path.join('lookupcsv', f'exp{i}')
        os.makedirs(folder, exist_ok=True)
        
        datasets = {
            'train.csv': [header] + train_set,
            'valid.csv': [header] + valid_set,
            'test.csv':  [header] + test_data
        }
        
        for name, content in datasets.items():
            with open(os.path.join(folder, name), 'w', newline='', encoding='utf-8') as f:
                csv.writer(f, quoting=csv.QUOTE_ALL).writerows(content)

        print(f"\n📂 [Experiment {i}] - Fold {fold_idx}")
        print(f"{'Stage':<10} | {'Total':<6} | {'CN(0)':<6} | {'MCI(1)':<6} | {'AD(2)':<6}")
        print("-" * 45)
        print(f"{'Train':<10} | {len(train_set):<6} | {train_counts[0]:<6} | {train_counts[1]:<6} | {train_counts[2]:<6}")
        print(f"{'Valid':<10} | {len(valid_set):<6} | {valid_counts[0]:<6} | {valid_counts[1]:<6} | {valid_counts[2]:<6}")
        print(f"{'Test':<10} | {len(test_data):<6} | {test_counts[0]:<6} | {test_counts[1]:<6} | {test_counts[2]:<6}")

    print("\n" + "="*50)
    print("✅ 所有實驗資料夾已生成完成。")