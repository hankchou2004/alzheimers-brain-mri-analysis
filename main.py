from models.model import CNN,ResNet18_3D # 🔑 引入你需要的架構
from models.model_wrapper import ModelWrapper
from utils.utils import read_json, data_split
from utils.plot import plot_training_history, plot_test_confusion_matrix
import glob
import os


def main():
    # 1. 載入設定
    config = read_json('configs/config.json')
    params = config['model'] 

    task_name = 'n'  # 建議從 config 讀取或手動設定

    # 2. 數據切分 (視需求開啟)
    #data_split(config['repeat_time'])

    for exp_idx in range(config['repeat_time']):
        print(f"\n=== Starting Experiment {exp_idx} ===")

        # 📂 構建權重搜尋路徑
        # 路徑範例: checkpoint_dir/ad_cn/CNN_exp0/CNN_best_metric_*.pth
        search_pattern = os.path.join(
            params['checkpoint_dir'], 
            task_name, 
            f"ResNet18_3D_exp{exp_idx}", 
            "ResNet18_3D_best_metric_*.pth"
        )
        
        # 尋找符合條件的檔案列表
        weight_files = glob.glob(search_pattern)
        
        pretrained_path = None
        if weight_files:
            # 如果有多個符合，通常取第一個或根據需求排序
            pretrained_path = weight_files[0]
            print(f"🚀 找到預訓練權重: {pretrained_path}")
        else:
            print(f"⚠️ 在 {search_pattern} 找不到權重，將從頭訓練。")
        
        # 2. 建立模型物件
        # 這樣 model.py 裡的類別名稱一旦改變，這裡的變數也會跟著動
        model_instance = ResNet18_3D
        # 3. 初始化 Wrapper
        wrapper = ModelWrapper(
            fil_num=params['fil_num'],
            drop_rate=params['drop_rate'],
            batch_size=params['batch_size'],
            balanced=params['balanced'],
            Data_dir=params['Data_dir'],
            exp_idx=exp_idx,#repeat time
            seed=1000,
            model=model_instance,         # 🔑 傳入實例
            pretrained_path=pretrained_path,
            metric='accuracy'
        )

        # 4. 訓練
        wrapper.train(lr=params['learning_rate'], epochs=params['train_epochs'])

        # 5. 測試與評估
        wrapper.test()

        # 在 main.py 中
        task_type = config.get('task', 'AD/CN') #支援 AD/CN, MCI/CN, AD/MCI, CN/Others, AD/Others, MCI/Others

        # --- 6. 繪圖區塊 ---
        print("📊 開始繪製實驗結果圖表...")

        # 1. 繪製訓練歷史曲線 (Loss/Acc 曲線)
        plot_training_history(wrapper.checkpoint_dir, wrapper.model_name, exp_idx)

        # 2. 繪製「最佳指標模型 (Best Metric)」的混淆矩陣
        # 預設 model_type 就是 'best_metric'
        plot_test_confusion_matrix(
            wrapper.checkpoint_dir, 
            wrapper.model_name, 
            exp_idx, 
            task=task_type,
            model_type='best_metric'
        )

        # 3. 繪製「最低損失模型 (Best Loss)」的混淆矩陣
        # 明確指定讀取 test_best_loss.csv
        plot_test_confusion_matrix(
            wrapper.checkpoint_dir, 
            wrapper.model_name, 
            exp_idx, 
            task=task_type,
            model_type='best_loss'
        )

        print(f"✅ 所有圖表已儲存至: {wrapper.checkpoint_dir}")

if __name__ == "__main__":
    main()


 