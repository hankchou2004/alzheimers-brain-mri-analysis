import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import sys


def _flood_fill_background(data: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """
    從 6 個角落進行 flood fill，將背景體素標記為 -10。
    條件：data < -0.6 且 temp < 0.8
    """
    new_data = data.copy()
    stack   = [(0,0,0),(180,0,0),(0,216,0),(180,216,0),
               (0,0,180),(180,0,180),(0,216,180),(180,216,180)]
    visited = set(stack)

    def valid(x, y, z):
        return 0 <= x < 181 and 0 <= y < 217 and 0 <= z < 181

    while stack:
        x, y, z = stack.pop()
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nx, ny, nz = x+dx, y+dy, z+dz
            if valid(nx, ny, nz) and (nx, ny, nz) not in visited \
               and data[nx, ny, nz] < -0.6 and temp[nx, ny, nz] < 0.8:
                visited.add((nx, ny, nz))
                new_data[nx, ny, nz] = -10
                stack.append((nx, ny, nz))

    return new_data


def remove_background(npy_path: str, temp: np.ndarray = None, save_jpg: bool = True) -> str:
    """
    對單一 .npy 檔案執行背景移除。

    Parameters
    ----------
    npy_path : str
        輸入的 .npy 檔案完整路徑（intensity normalization 後的輸出）。
    temp : np.ndarray, optional
        腦區遮罩陣列。若為 None，自動從同目錄的 brain_region.npy 載入。
    save_jpg : bool
        是否儲存三視圖 JPG 預覽圖（預設 True）。

    Returns
    -------
    str
        輸出 .npy 檔案的完整路徑（覆寫原檔或存至 back_removed/ 子目錄）。
    """
    npy_path = os.path.abspath(npy_path)

    # 自動載入 brain_region.npy（從 Preprocess 資料夾）
    if temp is None:
        brain_region_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brain_region.npy')
        if not os.path.exists(brain_region_path):
            raise FileNotFoundError(f"找不到 brain_region.npy: {brain_region_path}")
        temp = np.load(brain_region_path)

    data     = np.load(npy_path, allow_pickle=True)
    new_data = _flood_fill_background(data, temp)

    # 儲存至 back_removed/ 子目錄（避免覆蓋 normalization 結果）
    parent   = os.path.dirname(npy_path)
    out_dir  = os.path.join(os.path.dirname(parent), "back_removed")
    os.makedirs(out_dir, exist_ok=True)

    filename = os.path.basename(npy_path)

    # 選用：儲存三視圖 JPG
    if save_jpg:
        plt.figure(figsize=(12, 4))
        plt.subplot(131); plt.imshow(new_data[100, :, :], cmap='gray'); plt.axis('off'); plt.title('Axial')
        plt.subplot(132); plt.imshow(new_data[:, 100, :], cmap='gray'); plt.axis('off'); plt.title('Sagittal')
        plt.subplot(133); plt.imshow(new_data[:, :, 100], cmap='gray'); plt.axis('off'); plt.title('Coronal')
        jpg_path = os.path.join(out_dir, filename.replace('.npy', '.jpg'))
        plt.savefig(jpg_path, bbox_inches='tight')
        plt.close()
        print(f"[BackRemove] 預覽圖已儲存: {jpg_path}")

    # 將 -10 還原為 -1（背景統一值），再存檔
    final_data = np.where(new_data == -10, -np.ones((181, 217, 181)), new_data).astype(np.float32)
    out_path   = os.path.join(out_dir, filename)
    np.save(out_path, final_data)

    print(f"[BackRemove] 完成: {out_path}")
    return out_path


def back_remove(file, temp, new_path):
    """原有的批次介面（保留備用）"""
    os.makedirs(new_path, exist_ok=True)
    data     = np.load(file, allow_pickle=True)
    new_data = _flood_fill_background(data, temp)

    filename = os.path.basename(file)
    plt.subplot(131); plt.imshow(new_data[100, :, :])
    plt.subplot(132); plt.imshow(new_data[:, 100, :])
    plt.subplot(133); plt.imshow(new_data[:, :, 100])
    plt.savefig(os.path.join(new_path, filename.replace('.npy', '.jpg')))
    plt.close()

    final_data = np.where(new_data == -10, -np.ones((181, 217, 181)), new_data).astype(np.float32)
    np.save(os.path.join(new_path, filename), final_data)
    print(f"成功處理並儲存: {filename}")


if __name__ == "__main__":
    folder     = sys.argv[1]
    out_folder = sys.argv[2]
    os.makedirs(out_folder, exist_ok=True)

    temp  = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brain_region.npy'))
    files = glob(os.path.join(folder, "*.npy"))
    print(f"找到 {len(files)} 個檔案，開始處理...")
    for file in files:
        back_remove(file, temp, out_folder)
