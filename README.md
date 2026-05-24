# 阿茲海默症腦部 MRI 分析

基於深度學習的阿茲海默症偵測系統，使用 3D 腦部 MRI 影像，涵蓋自動化前處理、模型訓練、Grad-CAM 視覺化，以及互動式推論介面。

---

## 專案概述

本專案針對結構性 MRI 掃描實作完整的阿茲海默症（AD）分類流程，使用 [ADNI](https://adni.loni.usc.edu/) 資料集。骨幹模型為預訓練的 **3D ResNet-18**，經微調用於二元分類（認知正常 vs. 阿茲海默症 / 輕度認知障礙），並搭配多層 **Grad-CAM** 熱力圖提升可解釋性。

**主要參考文獻：**
- 前處理流程與專案架構部分參考自 [vkola-lab/brain2020](https://github.com/vkola-lab/brain2020)（MIT 授權）。
- 3D ResNet-18 模型架構參考自：*Learning-Based Progression Detection of Alzheimer's Disease Using 3D MRI Images*，International Journal of Intelligent Systems，2025。[https://doi.org/10.1155/int/3981977](https://doi.org/10.1155/int/3981977)

---

## 專案結構

```
.
├── configs/
│   └── config.json              # 訓練超參數與路徑設定
├── data/
│   └── dataloader.py            # 含 3D 資料增強的資料集類別
├── models/
│   ├── model.py                 # CNN 與 ResNet18_3D 架構
│   └── model_wrapper.py         # 訓練 / 評估封裝
├── utils/
│   ├── utils.py                 # CSV 讀取、指標計算、K-Fold 分割
│   └── plot.py                  # 損失/準確率曲線與混淆矩陣
├── Preprocess/
│   ├── __init__.py
│   ├── registration.py          # FSL FLIRT 配準（透過 WSL）
│   ├── registration.sh          # 完整配準流程的 Shell 腳本
│   ├── registration0427.py      # 支援模式選擇的擴充配準腳本
│   ├── registration0427.sh      # 支援 full / no_bet / no_reg 模式的 Shell 腳本
│   ├── intensity_normalization_and_clip.py  # Z-score 正規化 + 截斷至 [-1, 2.5]
│   ├── back_remove.py           # 洪水填充背景移除
│   └── brain_region.npy         # 背景移除用的腦部遮罩
├── lookupcsv/
│   ├── ADNI.csv                 # 完整受試者詮釋資料
│   └── exp{N}/                  # 各實驗的訓練 / 驗證 / 測試分割
│       ├── train.csv
│       ├── valid.csv
│       └── test.csv
├── main.py                      # 訓練入口
├── main_gradcam.py              # 批次 Grad-CAM 生成
├── preprocess.py                # CLI 前處理工具
├── inference_demo.py            # 互動式介面（NeuroScan AI）
├── gradcam_triptych.py          # 互動式多受試者 Grad-CAM 檢視器
├── gradcam_to_png.py            # 將中間切片熱力圖匯出為 PNG
└── convert_to_nii.bat           # DICOM → NIfTI 批次轉換（Windows）
```

---

## 環境需求

- Python ≥ 3.10
- PyTorch ≥ 2.0（含 CUDA）
- FSL（MRI 配準使用，Windows 上需安裝於 WSL）
- 主要 Python 套件：

```bash
pip install torch torchvision nibabel numpy scipy matplotlib seaborn \
            scikit-learn tqdm pandas customtkinter
```

---

## 資料準備

### 步驟一 — 將 DICOM 轉換為 NIfTI（Windows）

編輯 `convert_to_nii.bat` 中的路徑，執行後即可透過 `dcm2niix` 批次轉換 DICOM 檔案。

### 步驟二 — NIfTI 檔案前處理

`preprocess.py` 是獨立的 CLI 工具，可執行完整前處理流程：

```bash
# 處理單一檔案
python preprocess.py --input path/to/scan.nii --output path/to/output/

# 遞迴處理整個資料夾
python preprocess.py --input path/to/nii_folder/ --output path/to/output/
```

流程依序執行三個階段：

1. **影像配準** — 使用 FSL FLIRT 線性配準至 MNI152 1mm 空間（Windows 透過 WSL 執行）。`registration0427.py` 提供三種模式：
   - `full` — 重新定向 → robustfov → BET 顱骨剝除 → FLIRT MNI 配準
   - `no_bet` — 重新定向 → robustfov → FLIRT MNI 配準（跳過顱骨剝除）
   - `no_reg` — 重新定向 → robustfov → BET 顱骨剝除（不執行配準）
2. **強度正規化** — Z-score 正規化後截斷至 `[-1, 2.5]`，輸出尺寸為 `(181, 217, 181)`。
3. **背景移除** — 從體積角落進行洪水填充，將非腦部體素設為 `-1`，以 `brain_region.npy` 作為引導遮罩。

### 步驟三 — 生成資料分割

編輯 `config.json`，並在 `main.py` 中取消 `data_split()` 的註解並執行，即可在 `lookupcsv/exp{N}/` 下生成分層 K-Fold 的 `train/valid/test` CSV 分割檔。

---

## 模型訓練

編輯 `configs/config.json`：

```json
{
    "repeat_time": 5,
    "model": {
        "fil_num": 20,
        "drop_rate": 0.3,
        "batch_size": 2,
        "balanced": 0,
        "Data_dir": "path/to/preprocessed/",
        "learning_rate": 0.0001,
        "train_epochs": 100,
        "checkpoint_dir": "path/to/checkpoints/"
    }
}
```

接著執行：

```bash
python main.py
```

對每個實驗索引，程式將依序：
1. 若指定路徑存在預訓練權重則載入。
2. 使用 **Adam** 優化器搭配 **ReduceLROnPlateau** 學習率排程及混合精度（AMP）訓練模型。
3. 分別依驗證準確率（`best_metric`）與驗證損失（`best_loss`）儲存最佳模型。
4. 以兩種最佳模型對訓練 / 驗證 / 測試集進行評估，並將結果匯出為 CSV。
5. 生成損失 / 準確率曲線與混淆矩陣（PNG 格式）。

---

## 模型架構

### ResNet18_3D

主要模型基於 `torchvision.models.video.r3d_18`（Kinetics 預訓練），針對單通道 3D MRI 進行調整：

- **輸入層** 由 3 通道 RGB 改為 1 通道灰階：`Conv3d(1, 64, kernel=(3,7,7), stride=(1,2,2))`。
- **分類頭** 替換為 `Dropout → Linear(512, 2)`。
- **前向傳播** 額外輸出中間特徵圖（`layer2`、`layer3`、`layer4`）以供 Grad-CAM 使用。

### CNN（基準模型）

輕量級自定義 3D CNN，包含四個 `ConvLayer` 區塊（Conv3d → BatchNorm → LeakyReLU → MaxPool3d → Dropout），後接兩層全連接層。

---

## Grad-CAM 視覺化

### 批次生成

```bash
python main_gradcam.py
```

在 `ResNet18_3D.gradcam_multi_layer/` 下為每位受試者建立資料夾，包含：
- `raw_mri.npy` — 前處理後的 MRI 體積
- `heatmap_layer2.npy`、`heatmap_layer3.npy`、`heatmap_layer4.npy` — Grad-CAM 激活圖
- `info.json` — 預測結果、信心分數，以及各層的梯度重要性

### 互動式檢視器

```bash
python gradcam_triptych.py
```

開啟 Matplotlib 介面，提供軸狀 / 冠狀 / 矢狀切面、層選擇器（layer2/3/4）、熱力圖開關，以及以方向鍵切換受試者等功能。

### 匯出為 PNG

編輯 `gradcam_to_png.py` 中的路徑後執行，可將中間切片熱力圖與原始 MRI 切片儲存為 PNG 影像。

---

## 互動式推論介面

```bash
python inference_demo.py
```

啟動 **NeuroScan AI**，一款深色主題桌面應用程式（CustomTkinter），支援：

- 載入已訓練的 `.pth` 權重檔
- 選擇單一 NIfTI 檔案或批次資料夾
- 選擇性載入 ADNI CSV 進行受試者詮釋資料查詢
- 在背景執行緒中執行完整前處理流程與 Grad-CAM 推論
- 顯示軸狀 / 冠狀 / 矢狀切面，可調整熱力圖疊加（透明度、閾值）
- 切換 layer2/3/4 激活圖
- 顯示預測結果、信心分數及各層梯度重要性長條圖

---

## 輸出檔案

| 檔案 | 說明 |
|------|------|
| `{model}_best_metric_{epoch}.pth` | 驗證準確率最高時的模型權重 |
| `{model}_best_loss_{epoch}.pth` | 驗證損失最低時的模型權重 |
| `{model}_training_log.csv` | 每個 Epoch 的訓練 / 驗證損失、準確率、學習率 |
| `{model}_test_best_metric.csv` | 最佳準確率模型的測試指標 |
| `{model}_test_best_loss.csv` | 最佳損失模型的測試指標 |
| `{model}_exp{N}_loss_curve.png` | 訓練損失曲線 |
| `{model}_exp{N}_accuracy_curve.png` | 訓練準確率曲線 |
| `{model}_exp{N}_{task}_{type}_cm.png` | 含指標的混淆矩陣 |

---

## 致謝

- MRI 前處理流程架構改編自 [vkola-lab/brain2020](https://github.com/vkola-lab/brain2020)（Kolachalama 實驗室，MIT 授權）。
- 3D ResNet-18 架構設計參考自：Henry Horng-Shing Lu 等人，*Learning-Based Progression Detection of Alzheimer's Disease Using 3D MRI Images*，International Journal of Intelligent Systems，2025。[https://doi.org/10.1155/int/3981977](https://doi.org/10.1155/int/3981977)
- MRI 資料來自 [阿茲海默症神經影像學倡議（ADNI）](https://adni.loni.usc.edu/)。
- 影像配準使用 [FSL FLIRT](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT)。

---

## 授權

本專案僅供學術與研究使用，詳見 `LICENSE`。

# Alzheimer's Brain MRI Analysis

Deep learning-based Alzheimer's disease detection using 3D brain MRI images, featuring automated preprocessing, model training, Grad-CAM visualization, and an interactive inference GUI.

---

## Overview

This project implements a full pipeline for Alzheimer's disease (AD) classification from structural MRI scans using the [ADNI](https://adni.loni.usc.edu/) dataset. The backbone is a pretrained **3D ResNet-18** fine-tuned for binary classification (CN vs. AD / MCI), with multi-layer **Grad-CAM** heatmaps for interpretability.

**Key references:**
- Preprocessing pipeline and project structure partially adapted from [vkola-lab/brain2020](https://github.com/vkola-lab/brain2020) (MIT License).
- 3D ResNet-18 model architecture referenced from: *Learning-Based Progression Detection of Alzheimer's Disease Using 3D MRI Images*, International Journal of Intelligent Systems, 2025. [https://doi.org/10.1155/int/3981977](https://doi.org/10.1155/int/3981977)

---

## Project Structure

```
.
├── configs/
│   └── config.json              # Training hyperparameters and paths
├── data/
│   └── dataloader.py            # Dataset class with 3D augmentation
├── models/
│   ├── model.py                 # CNN and ResNet18_3D architectures
│   └── model_wrapper.py         # Training / evaluation wrapper
├── utils/
│   ├── utils.py                 # CSV reading, metrics, K-Fold split
│   └── plot.py                  # Loss/accuracy curves and confusion matrix
├── Preprocess/
│   ├── __init__.py
│   ├── registration.py          # FSL FLIRT registration (via WSL)
│   ├── registration.sh          # Shell script for full registration pipeline
│   ├── registration0427.py      # Extended registration with mode selection
│   ├── registration0427.sh      # Shell script supporting full / no_bet / no_reg modes
│   ├── intensity_normalization_and_clip.py  # Z-score normalization + clip to [-1, 2.5]
│   ├── back_remove.py           # Flood-fill background removal
│   └── brain_region.npy         # Brain mask for background removal
├── lookupcsv/
│   ├── ADNI.csv                 # Full subject metadata
│   └── exp{N}/                  # train / valid / test splits per experiment
│       ├── train.csv
│       ├── valid.csv
│       └── test.csv
├── main.py                      # Training entry point
├── main_gradcam.py              # Batch Grad-CAM generation
├── preprocess.py                # CLI preprocessing tool
├── inference_demo.py            # Interactive GUI (NeuroScan AI)
├── gradcam_triptych.py          # Interactive multi-subject Grad-CAM viewer
├── gradcam_to_png.py            # Export middle-slice heatmaps to PNG
└── convert_to_nii.bat           # DICOM → NIfTI batch conversion (Windows)
```

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 with CUDA
- FSL (for MRI registration, installed inside WSL on Windows)
- Key Python packages:

```bash
pip install torch torchvision nibabel numpy scipy matplotlib seaborn \
            scikit-learn tqdm pandas customtkinter
```

---

## Data Preparation

### Step 1 — Convert DICOM to NIfTI (Windows)

Edit paths in `convert_to_nii.bat`, then run it to batch-convert DICOM files using `dcm2niix`.

### Step 2 — Preprocess NIfTI files

`preprocess.py` is a standalone CLI tool that runs the full preprocessing pipeline:

```bash
# Process a single file
python preprocess.py --input path/to/scan.nii --output path/to/output/

# Process an entire folder (recursive)
python preprocess.py --input path/to/nii_folder/ --output path/to/output/
```

The pipeline runs three stages in sequence:

1. **Registration** — FSL FLIRT linear registration to MNI152 1mm space (via WSL on Windows). Three modes are available via `registration0427.py`:
   - `full` — reorient → robustfov → BET skull stripping → FLIRT MNI registration
   - `no_bet` — reorient → robustfov → FLIRT MNI registration (skip skull stripping)
   - `no_reg` — reorient → robustfov → BET skull stripping only (no registration)
2. **Intensity normalization** — Z-score normalization followed by clipping to `[-1, 2.5]`, output shape `(181, 217, 181)`.
3. **Background removal** — Flood-fill from volume corners to set non-brain voxels to `-1`, guided by `brain_region.npy`.

### Step 3 — Generate data splits

Edit `config.json` and run `data_split()` inside `main.py` (uncomment the call) to generate stratified K-Fold `train/valid/test` CSV splits under `lookupcsv/exp{N}/`.

---

## Training

Edit `configs/config.json`:

```json
{
    "repeat_time": 5,
    "model": {
        "fil_num": 20,
        "drop_rate": 0.3,
        "batch_size": 2,
        "balanced": 0,
        "Data_dir": "path/to/preprocessed/",
        "learning_rate": 0.0001,
        "train_epochs": 100,
        "checkpoint_dir": "path/to/checkpoints/"
    }
}
```

Then run:

```bash
python main.py
```

This will, for each experiment index:
1. Load pretrained weights if found at the expected checkpoint path.
2. Train the model using **Adam** optimizer with **ReduceLROnPlateau** scheduling and mixed-precision (AMP).
3. Save the best model by validation accuracy (`best_metric`) and by validation loss (`best_loss`).
4. Evaluate both saved models on train/valid/test sets and export results to CSV.
5. Generate loss/accuracy curves and confusion matrices as PNG files.

---

## Model Architecture

### ResNet18_3D

The primary model is based on `torchvision.models.video.r3d_18` (Kinetics pretrained), adapted for single-channel 3D MRI:

- **Input layer** modified from 3-channel RGB to 1-channel grayscale: `Conv3d(1, 64, kernel=(3,7,7), stride=(1,2,2))`.
- **Classification head** replaced with `Dropout → Linear(512, 2)`.
- **Forward pass** exposes intermediate feature maps (`layer2`, `layer3`, `layer4`) for Grad-CAM.

### CNN (baseline)

A lightweight custom 3D CNN with four `ConvLayer` blocks (Conv3d → BatchNorm → LeakyReLU → MaxPool3d → Dropout) followed by two dense layers.

---

## Grad-CAM Visualization

### Batch generation

```bash
python main_gradcam.py
```

Generates per-subject folders under `ResNet18_3D.gradcam_multi_layer/` containing:
- `raw_mri.npy` — preprocessed MRI volume
- `heatmap_layer2.npy`, `heatmap_layer3.npy`, `heatmap_layer4.npy` — Grad-CAM activation maps
- `info.json` — prediction, confidence, and per-layer gradient importance

### Interactive viewer

```bash
python gradcam_triptych.py
```

Opens a Matplotlib GUI with axial/coronal/sagittal views, a layer selector (layer2/3/4), heatmap toggle, and subject navigation via arrow keys.

### Export to PNG

Edit paths in `gradcam_to_png.py` and run it to save middle-slice heatmaps and the raw MRI slice as PNG images.

---

## Interactive Inference GUI

```bash
python inference_demo.py
```

Launches **NeuroScan AI**, a dark-themed desktop application (CustomTkinter) that supports:

- Loading a trained `.pth` weight file
- Selecting a single NIfTI file or a batch folder
- Optionally loading an ADNI CSV for subject metadata lookup
- Running the full preprocessing pipeline and Grad-CAM inference in a background thread
- Displaying axial/coronal/sagittal views with adjustable heatmap overlay (opacity, threshold)
- Switching between layer2/3/4 activation maps
- Showing prediction, confidence score, and per-layer gradient importance bars

---

## Outputs

| File | Description |
|------|-------------|
| `{model}_best_metric_{epoch}.pth` | Weights at highest validation accuracy |
| `{model}_best_loss_{epoch}.pth` | Weights at lowest validation loss |
| `{model}_training_log.csv` | Epoch-level train/valid loss, accuracy, LR |
| `{model}_test_best_metric.csv` | Test metrics for best-metric model |
| `{model}_test_best_loss.csv` | Test metrics for best-loss model |
| `{model}_exp{N}_loss_curve.png` | Training loss curves |
| `{model}_exp{N}_accuracy_curve.png` | Training accuracy curves |
| `{model}_exp{N}_{task}_{type}_cm.png` | Confusion matrix with metrics |

---

## Acknowledgements

- MRI preprocessing pipeline structure adapted from [vkola-lab/brain2020](https://github.com/vkola-lab/brain2020) (Kolachalama Laboratory, MIT License).
- 3D ResNet-18 architecture design informed by: Henry Horng-Shing Lu et al., *Learning-Based Progression Detection of Alzheimer's Disease Using 3D MRI Images*, International Journal of Intelligent Systems, 2025. [https://doi.org/10.1155/int/3981977](https://doi.org/10.1155/int/3981977)
- MRI data from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/).
- Registration performed using [FSL FLIRT](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT).

---

## License

This project is for academic and research use. See `LICENSE` for details.