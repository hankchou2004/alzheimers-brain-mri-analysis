import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import ast


def plot_training_history(checkpoint_dir, model_name, exp_idx, save_dir=None,model_type='training_log'):
    """
    繪製 Loss 與 Accuracy 曲線（train() 輸出的 {model_name}_training_log.csv）
    只畫 train/valid loss 和 train/valid accuracy，不畫 metric
    """
    csv_name = f'{model_name}_{model_type}.csv'
    csv_path = os.path.join(checkpoint_dir, csv_name)

    if not os.path.exists(csv_path):
        print(f"Error: {csv_name} not found at {csv_path}")
        return

    # 讀 CSV（有 header）
    df = pd.read_csv(csv_path)

    if save_dir is None:
        save_dir = os.path.dirname(csv_path)
    os.makedirs(save_dir, exist_ok=True)

    # ===== Loss 曲線 =====
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
    plt.plot(df['epoch'], df['valid_loss'], label='Valid Loss', color='red', linestyle='--')
    plt.title(f'{model_name} Loss (Exp {exp_idx})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_fig_path = os.path.join(save_dir, f'{model_name}_exp{exp_idx}_loss_curve.png')
    plt.savefig(loss_fig_path)
    plt.close()
    print(f"Loss curve saved to: {loss_fig_path}")

    # ===== Accuracy 曲線 =====
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy', color='green')
    plt.plot(df['epoch'], df['valid_acc'], label='Valid Accuracy', color='orange', linestyle='--')
    plt.title(f'{model_name} Accuracy (Exp {exp_idx})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    acc_fig_path = os.path.join(save_dir, f'{model_name}_exp{exp_idx}_accuracy_curve.png')
    plt.savefig(acc_fig_path)
    plt.close()
    print(f"Accuracy curve saved to: {acc_fig_path}")


def plot_test_confusion_matrix(
    checkpoint_dir,
    model_name,
    exp_idx,
    task='AD/CN',       # 支援 AD/CN, MCI/CN, AD/MCI, CN/Others, AD/Others, MCI/Others
    model_type='best_metric',
    save_dir=None
):
    """
    從 test CSV 繪製 confusion matrix，並根據 task 自動調整類別名稱
    """
    # 1. 根據 task 定義標籤 (新增 OvR 支援)
    # 格式：{task_name: [Label_0, Label_1]}
    # 注意：Label_0 通常代表負樣本 (Others/NL)，Label_1 代表正樣本 (Target)
    label_dict = {
        # --- 原有二元分類 ---
        'AD/CN':   ['CN', 'AD'],
        'MCI/CN':  ['CN', 'MCI'],
        'AD/MCI':  ['MCI', 'AD'],
        
        # --- 新增 OvR 分類 (與 read_csv 的 label_map 對應) ---
        # 根據你 read_csv 的邏輯：Target 為 1, Others 為 0
        'CN/Others':  ['Others', 'CN'],
        'AD/Others':  ['Others', 'AD'],
        'MCI/Others': ['Others', 'MCI']
    }
    
    # 取得對應標籤
    labels = label_dict.get(task, ['Class 0', 'Class 1'])

    csv_name = f'{model_name}_test_{model_type}.csv'
    csv_path = os.path.join(checkpoint_dir, csv_name)

    if not os.path.exists(csv_path):
        print(f"Error: {csv_name} not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df_test = df[df['stage'] == 'test']
    
    if len(df_test) == 0:
        print("Error: no test stage found in CSV")
        return

    row = df_test.iloc[0]

    # --- 解析混淆矩陣 ---
    try:
        cm = ast.literal_eval(row['confusion_matrix'])
        cm = np.array(cm)
    except Exception as e:
        print("Error parsing confusion_matrix:", e)
        return

    # 指標讀取
    acc         = float(row['acc'])
    precision   = float(row['precision'])
    recall      = float(row['recall'])
    f1          = float(row['f1'])
    specificity = float(row['specificity'])

    if save_dir is None:
        save_dir = checkpoint_dir
    os.makedirs(save_dir, exist_ok=True)

    # --- 混淆矩陣排列 ---
    # 結構為 [[TN, FP], [FN, TP]]
    # 繪圖：y 軸是 True，x 軸是 Pred
    # corrected_cm[0] 代表 True Others/CN, corrected_cm[1] 代表 True Target
    corrected_cm = np.array([
        [cm[0][0], cm[1][0]], # Row 0: True Negative (Label 0), False Positive (Label 1)
        [cm[0][1], cm[1][1]]  # Row 1: False Negative (Label 0), True Positive (Label 1)
    ])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corrected_cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[f'Pred {labels[0]}', f'Pred {labels[1]}'],
        yticklabels=[f'True {labels[0]}', f'True {labels[1]}'],
        annot_kws={"size": 14}
    )

    # 標題加入 Task 資訊
    plt.title(
        f'{model_name} ({task}, Exp {exp_idx})\n'
        f'Acc={acc:.3f}, Prec={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}\n'
        f'Spec={specificity:.3f}'
    )

    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.tight_layout()

    # 存檔名稱區分 task (將斜線換成底線防止路徑錯誤)
    task_str = task.replace('/', '_')
    save_path = os.path.join(
        save_dir,
        f'{model_name}_exp{exp_idx}_{task_str}_{model_type}_cm.png'
    )
    plt.savefig(save_path)
    plt.close()

    print(f"✅ {task} confusion matrix saved to: {save_path}")