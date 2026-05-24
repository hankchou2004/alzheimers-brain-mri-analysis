import nibabel as nib
import numpy as np
from glob import glob
from subprocess import call
import sys
import os

# nipype 只在備用的 registration() 函數中才需要，
# 且在 Python 3.13 / 某些 Windows 環境下可能無法載入，
# 所以改成延遲 import，不在模組頂層載入。

# ── 合法的 mode 值 ────────────────────────────────────────────────────────────
VALID_MODES = ("full", "no_reg", "no_bet")
"""
mode 說明
---------
full    完整流程：reorient → robustfov → bet 去頭骨 → FLIRT 配準 MNI  （預設）
no_reg  無 MNI 配準：reorient → robustfov → bet 去頭骨（輸出去頭骨後影像）
no_bet  無去頭骨：reorient → robustfov → FLIRT 配準 MNI（跳過 bet）
"""


def to_wsl_path(win_path):
    """將 Windows 路徑轉換為 WSL 可識別的 Linux 路徑"""
    path = os.path.abspath(win_path)
    drive, rest = os.path.splitdrive(path)
    if drive:
        drive_letter = drive[0].lower()
        return f"/mnt/{drive_letter}{rest.replace(chr(92), '/')}"
    return path.replace(chr(92), '/')


import nibabel as nib
import numpy as np
from glob import glob
from subprocess import call
import sys
import os

# nipype 只在備用的 registration() 函數中才需要，
# 且在 Python 3.13 / 某些 Windows 環境下可能無法載入，
# 所以改成延遲 import，不在模組頂層載入。

# ── 合法的 mode 值 ────────────────────────────────────────────────────────────
VALID_MODES = ("full", "no_reg", "no_bet")
"""
mode 說明
---------
full    完整流程：reorient → robustfov → bet 去頭骨 → FLIRT 配準 MNI  （預設）
no_reg  無 MNI 配準：reorient → robustfov → bet 去頭骨（輸出去頭骨後影像）
no_bet  無去頭骨：reorient → robustfov → FLIRT 配準 MNI（跳過 bet）
"""


def to_wsl_path(win_path):
    """將 Windows 路徑轉換為 WSL 可識別的 Linux 路徑"""
    path = os.path.abspath(win_path)
    drive, rest = os.path.splitdrive(path)
    if drive:
        drive_letter = drive[0].lower()
        return f"/mnt/{drive_letter}{rest.replace(chr(92), '/')}"
    return path.replace(chr(92), '/')


def run_registration(nifti_path: str, mode: str = "full", out_folder: str = None) -> str:
    """
    對單一 NIfTI 檔案執行 FSL registration（透過 WSL bash pipeline）。

    Parameters
    ----------
    nifti_path : str
        輸入的 .nii 或 .nii.gz 檔案的完整 Windows 路徑。
    mode : str, optional
        處理模式，可選值：
            "full"    完整流程（預設）：去頭骨 + MNI 配準
            "no_reg"  無 MNI 配準：僅做 reorient / robustfov / bet 去頭骨
            "no_bet"  無去頭骨：reorient / robustfov 後直接 FLIRT 配準 MNI

    Returns
    -------
    str
        Registration 完成後，輸出 .nii 檔案的完整 Windows 路徑。

    Raises
    ------
    ValueError
        mode 不在合法清單中。
    RuntimeError
        Shell script 回傳非零 exit code。
    FileNotFoundError
        輸出檔案不存在（shell script 未成功產生結果）。
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"不合法的 mode：'{mode}'。請使用 {VALID_MODES} 其中之一。"
        )

    nifti_path = os.path.abspath(nifti_path)
    filename   = os.path.basename(nifti_path)
    raw_folder = os.path.dirname(nifti_path)

    # ↓ 改這裡：優先用傳入的 out_folder，否則才用預設位置
    if out_folder is None:
        out_folder = os.path.join(os.path.dirname(raw_folder), f"registration_{mode}")
        
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, "tmp"), exist_ok=True)

    # 轉換為 WSL 路徑（確保結尾有斜線）
    raw_linux = to_wsl_path(raw_folder).rstrip('/') + '/'
    out_linux = to_wsl_path(out_folder).rstrip('/') + '/'

    # registration.sh 的位置（與本檔案同目錄）
    sh_dir      = os.path.dirname(os.path.abspath(__file__))
    sh_wsl_path = to_wsl_path(os.path.join(sh_dir, "registration0427.sh"))

    # 將 mode 作為第四個引數傳給 shell script
    cmd = f'wsl bash "{sh_wsl_path}" "{raw_linux}" "{filename}" "{out_linux}" "{mode}"'
    print(f"[Registration] mode={mode}  執行指令: {cmd}")
    ret = call(cmd, shell=True)
    if ret != 0:
        raise RuntimeError(
            f"Registration 失敗（return code {ret}，mode={mode}）：{filename}"
        )

    out_path = os.path.join(out_folder, filename)
    if not os.path.exists(out_path):
        raise FileNotFoundError(
            f"Registration 後找不到輸出檔案（mode={mode}）: {out_path}"
        )

    print(f"[Registration] 完成（mode={mode}）: {out_path}")
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
    import argparse

    parser = argparse.ArgumentParser(
        description="FSL Registration 單檔測試入口"
    )
    parser.add_argument("nifti_path", help="輸入 .nii 或 .nii.gz 檔案路徑")
    parser.add_argument(
        "--mode", "-m",
        choices=VALID_MODES,
        default="full",
        help=(
            "處理模式（預設：full）\n"
            "  full    完整流程（去頭骨 + MNI 配準）\n"
            "  no_reg  無 MNI 配準（僅去頭骨）\n"
            "  no_bet  無去頭骨（直接 MNI 配準）"
        ),
    )
    args = parser.parse_args()
    result = run_registration(args.nifti_path, mode=args.mode)
    print(f"輸出: {result}")


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
    import argparse

    parser = argparse.ArgumentParser(
        description="FSL Registration 單檔測試入口"
    )
    parser.add_argument("nifti_path", help="輸入 .nii 或 .nii.gz 檔案路徑")
    parser.add_argument(
        "--mode", "-m",
        choices=VALID_MODES,
        default="full",
        help=(
            "處理模式（預設：full）\n"
            "  full    完整流程（去頭骨 + MNI 配準）\n"
            "  no_reg  無 MNI 配準（僅去頭骨）\n"
            "  no_bet  無去頭骨（直接 MNI 配準）"
        ),
    )
    args = parser.parse_args()
    result = run_registration(args.nifti_path, mode=args.mode)
    print(f"輸出: {result}")