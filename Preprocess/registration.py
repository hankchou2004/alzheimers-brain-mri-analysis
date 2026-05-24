import nibabel as nib
import numpy as np
from glob import glob
from subprocess import call
import sys
import os

# nipype 只在備用的 registration() 函數中才需要，
# 且在 Python 3.13 / 某些 Windows 環境下可能無法載入，
# 所以改成延遲 import，不在模組頂層載入。


def to_wsl_path(win_path):
    """將 Windows 路徑轉換為 WSL 可識別的 Linux 路徑"""
    path = os.path.abspath(win_path)
    drive, rest = os.path.splitdrive(path)
    if drive:
        drive_letter = drive[0].lower()
        return f"/mnt/{drive_letter}{rest.replace(chr(92), '/')}"
    return path.replace(chr(92), '/')


def run_registration(nifti_path: str) -> str:
    """
    對單一 NIfTI 檔案執行 FSL registration（透過 WSL bash pipeline）。

    Parameters
    ----------
    nifti_path : str
        輸入的 .nii 或 .nii.gz 檔案的完整 Windows 路徑。

    Returns
    -------
    str
        Registration 完成後，輸出 .nii 檔案的完整 Windows 路徑。
    """
    nifti_path = os.path.abspath(nifti_path)
    filename   = os.path.basename(nifti_path)
    raw_folder = os.path.dirname(nifti_path)

    # 輸出資料夾：在原始資料夾旁建立 registration/ 子目錄
    out_folder = os.path.join(os.path.dirname(raw_folder), "registration")
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, "tmp"), exist_ok=True)

    # 轉換為 WSL 路徑（確保結尾有斜線）
    raw_linux = to_wsl_path(raw_folder).rstrip('/') + '/'
    out_linux = to_wsl_path(out_folder).rstrip('/') + '/'

    # registration.sh 的位置（與本檔案同目錄）
    sh_dir      = os.path.dirname(os.path.abspath(__file__))
    sh_wsl_path = to_wsl_path(os.path.join(sh_dir, "registration.sh"))

    cmd = f'wsl bash "{sh_wsl_path}" "{raw_linux}" "{filename}" "{out_linux}"'
    print(f"[Registration] 執行指令: {cmd}")
    ret = call(cmd, shell=True)
    if ret != 0:
        raise RuntimeError(f"Registration 失敗（return code {ret}）：{filename}")

    out_path = os.path.join(out_folder, filename)
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"Registration 後找不到輸出檔案: {out_path}")

    print(f"[Registration] 完成: {out_path}")
    return out_path


def registration(in_file, out_file, reference):
    """原有的 nipype FLIRT 單檔介面（保留備用）"""
    from nipype.interfaces import fsl          # 延遲 import，避免啟動時就爆炸
    fsl.FSLCommand.set_default_output_type('NIFTI')
    flt = fsl.FLIRT(bins=640, cost_func='mutualinfo')
    flt.inputs.in_file   = in_file
    flt.inputs.out_file  = out_file
    flt.inputs.reference = reference
    flt.run()


if __name__ == "__main__":
    raw_folder = r"D:\0214_adni1_nii\cn\nii"
    out_folder = r"D:\0214_adni1_nii\cn\registration"
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, "tmp"), exist_ok=True)

    raw_linux = to_wsl_path(raw_folder).rstrip('/') + '/'
    out_linux = to_wsl_path(out_folder).rstrip('/') + '/'
    sh_dir    = os.path.dirname(os.path.abspath(__file__))
    sh_wsl    = to_wsl_path(os.path.join(sh_dir, "registration.sh"))

    for fullPath in glob(os.path.join(raw_folder, '*.nii')):
        filename = os.path.basename(fullPath)
        print(f"Processing: {filename}")
        cmd = f'wsl bash "{sh_wsl}" "{raw_linux}" "{filename}" "{out_linux}"'
        call(cmd, shell=True)
