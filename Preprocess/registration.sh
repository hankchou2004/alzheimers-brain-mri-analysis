#!/bin/sh
# This bash script use fsl to process brain MRI into MNI template

export FSLDIR=/home/hank/fsl
. ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:$PATH
export FSLOUTPUTTYPE='NIFTI'

# $1: raw folder, $2: filename, $3: output folder
echo "processing file from $1: $2"

# 建立暫存區並複製原始檔案 (加上雙引號防止路徑空格出錯)
mkdir -p "$3/tmp"
cp "$1/$2" "$3/tmp/$2"
cd "$3/tmp" || exit

# STEP 1: 方向轉正
${FSLDIR}/bin/fslreorient2std "$2" T1.nii

# STEP 2: 估計穩健視野
line=$(${FSLDIR}/bin/robustfov -i T1.nii | grep -v Final | head -n 1)

# 解析座標
x1=$(echo ${line} | awk '{print $1}'); x2=$(echo ${line} | awk '{print $2}')
y1=$(echo ${line} | awk '{print $3}'); y2=$(echo ${line} | awk '{print $4}')
z1=$(echo ${line} | awk '{print $5}'); z2=$(echo ${line} | awk '{print $6}')

# 四捨五入座標 (修正 printf 語法：移除逗號)
x1=$(printf "%.0f" $x1); x2=$(printf "%.0f" $x2)
y1=$(printf "%.0f" $y1); y2=$(printf "%.0f" $y2)
z1=$(printf "%.0f" $z1); z2=$(printf "%.0f" $z2)

# STEP 3: 裁切 ROI
${FSLDIR}/bin/fslmaths T1.nii -roi $x1 $x2 $y1 $y2 $z1 $z2 0 1 T1_roi.nii

# STEP 4: 去除頭蓋骨
${FSLDIR}/bin/bet T1_roi.nii T1_brain.nii -R

# STEP 5: 估計對齊矩陣 (Register to MNI)
# 建議加入全角度搜尋以防止對齊失敗
MNI_REF="${FSLDIR}/data/standard/MNI152_T1_1mm_brain"
${FSLDIR}/bin/flirt -in T1_brain.nii -ref "$MNI_REF" -omat orig_to_MNI.mat -searchrx -180 180 -searchry -180 180 -searchrz -180 180

# STEP 6: 套用矩陣到轉正後的原始影像
${FSLDIR}/bin/flirt -in T1.nii -ref "$MNI_REF" -applyxfm -init orig_to_MNI.mat -out T1_MNI.nii

# --- 重要：將結果移動到輸出資料夾 ---
if [ -f T1_MNI.nii ]; then
    mv T1_MNI.nii "$3/$2"
    echo "Saved: $3/$2"
else
    echo "Error: T1_MNI.nii was not created."
fi

# 清理暫存檔
rm -f "$3/tmp/"*