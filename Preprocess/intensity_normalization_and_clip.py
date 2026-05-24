import numpy as np
import nibabel as nib
import sys
import os
from glob import glob


def nifti_to_numpy(file: str) -> np.ndarray:
    """載入 NIfTI 並 pad/裁切至固定形狀 (181, 217, 181)"""
    img = nib.load(file).get_fdata()
    target_shape = (181, 217, 181)
    data = np.zeros(target_shape)
    x_lim = min(img.shape[0], 181)
    y_lim = min(img.shape[1], 217)
    z_lim = min(img.shape[2], 181)
    data[:x_lim, :y_lim, :z_lim] = img[:x_lim, :y_lim, :z_lim]
    return data


def normalization(scan: np.ndarray) -> np.ndarray:
    """Z-score 正規化"""
    return (scan - np.mean(scan)) / np.std(scan)


def clip(scan: np.ndarray) -> np.ndarray:
    """將數值限縮至 [-1, 2.5]"""
    return np.clip(scan, -1, 2.5)


def convert_to_npy(nifti_path: str) -> str:
    """
    將單一 NIfTI 檔案轉換為正規化並截斷後的 .npy 檔案。

    Parameters
    ----------
    nifti_path : str
        輸入的 .nii 檔案完整路徑（registration 後的輸出）。

    Returns
    -------
    str
        輸出 .npy 檔案的完整路徑（與輸入同目錄，副檔名改為 .npy）。
    """
    nifti_path = os.path.abspath(nifti_path)

    # 去除 .nii 或 .nii.gz 副檔名
    if nifti_path.endswith('.nii.gz'):
        npy_path = nifti_path[:-7] + '.npy'
    else:
        npy_path = os.path.splitext(nifti_path)[0] + '.npy'

    data = nifti_to_numpy(nifti_path)
    data = normalization(data)
    data = clip(data)
    np.save(npy_path, data)

    print(f"[Normalization] 完成: {npy_path}")
    return npy_path


if __name__ == "__main__":
    folder = sys.argv[1]
    for file in glob(os.path.join(folder, '*.nii')):
        convert_to_npy(file)
