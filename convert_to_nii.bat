@echo off
chcp 65001
setlocal enabledelayedexpansion

set SRC_DIR=C:\Users\User\輸入資料夾路徑
set OUT_DIR=C:\Users\User\輸出資料夾路徑
set DCM2NIIX=C:\Users\User\dcm2niix.exe

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo --------------------------------------------------
echo 開始處理 ADNI 深度路徑 (輸出純 .nii)...
echo --------------------------------------------------

for /r "%SRC_DIR%" %%d in (.) do (
    
    :: 1. 處理 DICOM 轉 nii (不壓縮)
    if exist "%%~fd\*.dcm" (
        echo [處理 DICOM] 目錄: %%~fd
        :: 修改點：-z n 表示不壓縮，輸出純 .nii
        "%DCM2NIIX%" -o "%OUT_DIR%" -z n -f "%%~nd" -r n "%%~fd"
    )

    :: 2. 處理原本就在子目錄的 .nii (直接複製)
    if exist "%%~fd\*.nii" (
        for %%f in ("%%~fd\*.nii") do (
            if not exist "%OUT_DIR%\%%~nxf" (
                echo [複製 NIfTI] 檔案: %%~nxf
                copy "%%~fd\%%~nxf" "%OUT_DIR%\"
            )
        )
    )
)

echo 全部處理完成！
pause