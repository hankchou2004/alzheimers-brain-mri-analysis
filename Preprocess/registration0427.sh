#!/bin/sh
# This bash script uses FSL to process brain MRI
#
# 使用方式:
#   bash registration.sh <raw_folder> <filename> <out_folder> [mode]
#
# mode 參數（可選，預設為 full）:
#   full        完整流程：reorient → robustfov → bet 去頭骨 → FLIRT 配準 MNI
#   no_reg      無 MNI 配準：reorient → robustfov → bet 去頭骨（輸出去頭骨後影像）
#   no_bet      無去頭骨：reorient → robustfov → FLIRT 配準 MNI（跳過 bet）

export FSLDIR=/home/hank/fsl
. ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:$PATH
export FSLOUTPUTTYPE='NIFTI'

# 引數
RAW_FOLDER="$1"
FILENAME="$2"
OUT_FOLDER="$3"
MODE="${4:-full}"   # 預設 full

echo "========================================"
echo "  FSL Registration Pipeline"
echo "  Mode    : $MODE"
echo "  File    : $FILENAME"
echo "  Input   : $RAW_FOLDER"
echo "  Output  : $OUT_FOLDER"
echo "========================================"

# 建立暫存區並複製原始檔案
# 建立暫存區並複製原始檔案
rm -rf "$OUT_FOLDER/tmp"          # ← 加這行，強制清空
mkdir -p "$OUT_FOLDER/tmp"
cp "$RAW_FOLDER/$FILENAME" "$OUT_FOLDER/tmp/$FILENAME"
cd "$OUT_FOLDER/tmp" || exit 1

MNI_REF="${FSLDIR}/data/standard/MNI152_T1_1mm_brain"

# ──────────────────────────────────────────
# STEP 1: 方向轉正（所有模式共用）
# ──────────────────────────────────────────
echo "[STEP 1] fslreorient2std"
${FSLDIR}/bin/fslreorient2std "$FILENAME" T1.nii

# ──────────────────────────────────────────
# STEP 2: 估計穩健視野（所有模式共用）
# ──────────────────────────────────────────
echo "[STEP 2] robustfov"
line=$(${FSLDIR}/bin/robustfov -i T1.nii | grep -v Final | head -n 1)

x1=$(echo ${line} | awk '{print $1}'); x2=$(echo ${line} | awk '{print $2}')
y1=$(echo ${line} | awk '{print $3}'); y2=$(echo ${line} | awk '{print $4}')
z1=$(echo ${line} | awk '{print $5}'); z2=$(echo ${line} | awk '{print $6}')

x1=$(printf "%.0f" $x1); x2=$(printf "%.0f" $x2)
y1=$(printf "%.0f" $y1); y2=$(printf "%.0f" $y2)
z1=$(printf "%.0f" $z1); z2=$(printf "%.0f" $z2)

# ──────────────────────────────────────────
# STEP 3: 裁切 ROI（所有模式共用）
# ──────────────────────────────────────────
echo "[STEP 3] fslmaths ROI crop"
${FSLDIR}/bin/fslmaths T1.nii -roi $x1 $x2 $y1 $y2 $z1 $z2 0 1 T1_roi.nii

# ──────────────────────────────────────────
# 依 MODE 分流
# ──────────────────────────────────────────

if [ "$MODE" = "no_reg" ]; then
    # ── 模式 2：去頭骨，不做 MNI 配準 ──────
    echo "[STEP 4] bet（去頭骨）"
    ${FSLDIR}/bin/bet T1_roi.nii T1_brain.nii -R

    echo "[SKIP] MNI registration（mode=no_reg）"

    RESULT="T1_brain.nii"

elif [ "$MODE" = "no_bet" ]; then
    # ── 模式 3：不去頭骨，直接配準 MNI ─────
    echo "[SKIP] bet（mode=no_bet）"

    echo "[STEP 4] flirt — 估計對齊矩陣（使用 T1_roi 做配準）"
    ${FSLDIR}/bin/flirt \
        -in T1_roi.nii -ref "$MNI_REF" \
        -omat orig_to_MNI.mat \
        -searchrx -180 180 -searchry -180 180 -searchrz -180 180

    echo "[STEP 5] flirt — 套用矩陣到方向轉正後影像"
    ${FSLDIR}/bin/flirt \
        -in T1.nii -ref "$MNI_REF" \
        -applyxfm -init orig_to_MNI.mat \
        -out T1_MNI.nii

    RESULT="T1_MNI.nii"

else
    # ── 模式 1：完整流程（預設 full）────────
    echo "[STEP 4] bet（去頭骨）"
    ${FSLDIR}/bin/bet T1_roi.nii T1_brain.nii -R

    echo "[STEP 5] flirt — 估計對齊矩陣"
    ${FSLDIR}/bin/flirt \
        -in T1_brain.nii -ref "$MNI_REF" \
        -omat orig_to_MNI.mat \
        -searchrx -180 180 -searchry -180 180 -searchrz -180 180

    echo "[STEP 6] flirt — 套用矩陣到方向轉正後影像"
    ${FSLDIR}/bin/flirt \
        -in T1.nii -ref "$MNI_REF" \
        -applyxfm -init orig_to_MNI.mat \
        -out T1_MNI.nii

    RESULT="T1_MNI.nii"
fi

# ──────────────────────────────────────────
# 輸出結果
# ──────────────────────────────────────────
if [ -f "$RESULT" ]; then
    mv "$RESULT" "$OUT_FOLDER/$FILENAME"
    echo "[Done] Saved: $OUT_FOLDER/$FILENAME"
else
    echo "[Error] 結果檔案未產生: $RESULT"
    exit 1
fi

# 清理暫存檔
rm -f "$OUT_FOLDER/tmp/"*