#!/usr/bin/env python3
r"""
preprocess.py  ─  獨立前處理工具
用法:
    # 處理單一檔案
    python preprocess.py --input path/to/scan.nii

    # 處理整個資料夾（遞迴搜尋 .nii / .nii.gz）
    python preprocess.py --input path/to/folder

    # 指定輸出目錄
    python preprocess.py --input path/to/folder --output path/to/out

    # 顯示說明
    python preprocess.py --help

    python preprocess.py --input "C:\Users\User\Desktop\demo\nii_data" --output "C:\Users\User\Desktop\demo\no_reg"
    python preprocess.py --input "C:\Users\User\Desktop\demo\nii_data" --output "C:\Users\User\Desktop\demo\no_bet"
    python preprocess.py --input "C:\Users\User\Desktop\demo\nii_data" --output "C:\Users\User\Desktop\demo\full"
"""

import os
import sys
import glob
import shutil
import argparse
import logging
from pathlib import Path

# ── 路徑設定（與原始專案相同）────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from Preprocess.registration0427                    import run_registration
from Preprocess.intensity_normalization_and_clip import convert_to_npy
from Preprocess.back_remove                     import remove_background

# ── 常數 ─────────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT = os.path.join(_SCRIPT_DIR, "output_preprocessed")

# ── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def collect_nii_files(input_path: str) -> list[str]:
    """接受單一檔案或資料夾，回傳所有 .nii / .nii.gz 路徑清單。"""
    p = Path(input_path)
    if p.is_file():
        if input_path.endswith((".nii", ".nii.gz")):
            return [str(p)]
        raise ValueError(f"不支援的檔案格式：{input_path}")
    if p.is_dir():
        files = (
            glob.glob(str(p / "**" / "*.nii"),    recursive=True) +
            glob.glob(str(p / "**" / "*.nii.gz"), recursive=True)
        )
        return sorted(files)
    raise FileNotFoundError(f"找不到路徑：{input_path}")


def preprocess_one(nii_path: str, output_dir: str) -> str | None:
    """
    對單一 NIfTI 執行完整前處理流程：
        1. Registration
        2. Intensity normalization & clip → .npy
        3. Background removal
    成功回傳輸出 .npy 路徑；失敗回傳 None。
    """
    sid = Path(nii_path).name.replace(".nii.gz", "").replace(".nii", "")
    dst = os.path.join(output_dir, sid + ".npy")

    try:
        log.info(f"[registration]   {sid}")
        # ↓ 加入 out_folder=output_dir，讓 registration 結果也存到指定位置
        reg = run_registration(nii_path, mode="no_bet", out_folder=os.path.join(output_dir, "registration"))

        log.info(f"[normalization]  {sid}")
        npy = convert_to_npy(reg)

        log.info(f"[back_remove]    {sid}")
        final = remove_background(npy)

        shutil.copy2(final, dst)
        log.info(f"[done]  → {dst}")
        return dst

    except Exception as e:
        log.error(f"[failed] {sid}: {e}")
        return None


def preprocess_all(nii_files: list[str], output_dir: str) -> dict:
    """
    批次處理並回傳結果摘要：
        { "success": [...], "failed": [...] }
    """
    os.makedirs(output_dir, exist_ok=True)
    total   = len(nii_files)
    success = []
    failed  = []

    for idx, nii_path in enumerate(nii_files, start=1):
        log.info(f"─── [{idx}/{total}] {os.path.basename(nii_path)}")
        result = preprocess_one(nii_path, output_dir)
        (success if result else failed).append(nii_path)

    return {"success": success, "failed": failed}


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="3D MRI 前處理：registration → normalization → background removal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="單一 .nii / .nii.gz 檔案，或含多個 NIfTI 的資料夾"
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT,
        help=f"輸出目錄（預設：{DEFAULT_OUTPUT}）"
    )
    args = parser.parse_args()

    # 收集檔案
    try:
        files = collect_nii_files(args.input)
    except (ValueError, FileNotFoundError) as e:
        log.error(str(e))
        sys.exit(1)

    if not files:
        log.warning("找不到任何 NIfTI 檔案，程式結束。")
        sys.exit(0)

    log.info(f"找到 {len(files)} 個檔案  →  輸出至 {args.output}")

    # 執行前處理
    summary = preprocess_all(files, args.output)

    # 結果摘要
    log.info("=" * 50)
    log.info(f"完成：{len(summary['success'])} 筆  |  失敗：{len(summary['failed'])} 筆")
    if summary["failed"]:
        log.warning("失敗清單：")
        for f in summary["failed"]:
            log.warning(f"  {f}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
