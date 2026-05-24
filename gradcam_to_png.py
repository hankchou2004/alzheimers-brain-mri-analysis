import numpy as np
import matplotlib.pyplot as plt
import os

def save_middle_slice_as_img(npy_path, output_name, is_heatmap=True):
    if not os.path.exists(npy_path):
        print(f"找不到檔案: {npy_path}")
        return

    data = np.load(npy_path)
    
    # 取得中間切片的索引
    mid_idx = data.shape[0] // 2
    slice_data = data[mid_idx]
    
    plt.figure(figsize=(6, 6))
    if is_heatmap:
        plt.imshow(slice_data, cmap='jet')
    else:
        plt.imshow(slice_data, cmap='gray')
        
    plt.axis('off')
    plt.savefig(output_name, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"已儲存: {output_name}")

# --- 修改這裡的路徑 ---
# 請確保路徑中的斜線是 / 或在字串前加 r
source_dir = r'C:\Users\User\Desktop\0303\ResNet18_3D.gradcam_multi_layer\I7025'
output_dir = r'C:\Users\User\Desktop\專題demo\0421' # 儲存到你現在寫程式的地方

# 1. 處理原始影像
raw_path = os.path.join(source_dir, 'raw_mri.npy')
save_middle_slice_as_img(raw_path, os.path.join(output_dir, 'raw_mri_mid.png'), is_heatmap=False)

# 2. 處理熱力圖
heatmaps = ['heatmap_layer2.npy', 'heatmap_layer3.npy', 'heatmap_layer4.npy']
for h in heatmaps:
    h_path = os.path.join(source_dir, h)
    output_png = os.path.join(output_dir, h.replace('.npy', '.png'))
    save_middle_slice_as_img(h_path, output_png, is_heatmap=True)