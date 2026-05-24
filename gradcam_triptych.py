import os
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.ndimage import zoom

class GradCamViewer:
    def __init__(self, base_path):
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"找不到路徑: {base_path}")
            
        self.subjects = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
        if not self.subjects:
            raise ValueError("指定的資料夾內沒有受試者數據。")
            
        self.base_path = base_path
        self.idx = 0
        self.show_heatmap = True 
        self.current_layer = 'layer4'  # 預設顯示最深層
        
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 9))
        plt.subplots_adjust(bottom=0.28, right=0.85, top=0.88, wspace=0.2)
        
        # 滑桿設定
        ax_s0 = plt.axes([0.15, 0.14, 0.45, 0.02], facecolor='lightgoldenrodyellow')
        ax_s1 = plt.axes([0.15, 0.10, 0.45, 0.02], facecolor='lightgoldenrodyellow')
        ax_s2 = plt.axes([0.15, 0.06, 0.45, 0.02], facecolor='lightgoldenrodyellow')
        
        self.slider0 = Slider(ax_s0, 'Axial   ', 0, 100, valinit=50, valfmt='%0.0f%%')
        self.slider1 = Slider(ax_s1, 'Coronal ', 0, 100, valinit=50, valfmt='%0.0f%%')
        self.slider2 = Slider(ax_s2, 'Sagittal', 0, 100, valinit=50, valfmt='%0.0f%%')
        
        # 🔑 切換層級的 RadioButtons (選取 Layer 2, 3, 4)
        ax_radio = plt.axes([0.65, 0.05, 0.10, 0.12], facecolor='#f0f0f0')
        self.radio = RadioButtons(ax_radio, ('layer2', 'layer3', 'layer4'), active=2)
        self.radio.on_clicked(self.change_layer)

        # 控制按鈕
        ax_btn = plt.axes([0.78, 0.08, 0.10, 0.05])
        self.btn = Button(ax_btn, 'Toggle Heatmap', color='lightgray', hovercolor='skyblue')
        self.btn.on_clicked(self.toggle_heatmap)

        self.cbar_ax = self.fig.add_axes([0.92, 0.35, 0.015, 0.4])
        self.cbar = None

        self.slider0.on_changed(self.update_plot)
        self.slider1.on_changed(self.update_plot)
        self.slider2.on_changed(self.update_plot)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.load_subject()

    def change_layer(self, label):
        """當點選 RadioButton 時觸發"""
        self.current_layer = label
        self.load_subject()  # 重新載入對應層級的 heatmap

    def toggle_heatmap(self, event):
        self.show_heatmap = not self.show_heatmap
        self.update_plot(None)

    def load_subject(self):
        sub_id = self.subjects[self.idx]
        path = os.path.join(self.base_path, sub_id)
        
        self.img = np.load(os.path.join(path, 'raw_mri.npy'))
        self.mask = self.img > (np.max(self.img) * 0.05)

        # 🔑 修改點：根據 current_layer 載入對應檔案
        heatmap_file = f'heatmap_{self.current_layer}.npy'
        heatmap_path = os.path.join(path, heatmap_file)
        
        if not os.path.exists(heatmap_path):
            # 如果找不到分層檔案，嘗試載入預設的 heatmap.npy
            heatmap_path = os.path.join(path, 'heatmap.npy')
            print(f"⚠️ 找不到 {heatmap_file}，載入預設 heatmap.npy")

        self.cam = np.load(heatmap_path)
        
        with open(os.path.join(path, 'info.json'), 'r') as f:
            self.info = json.load(f)
            
        scale_factor = np.array(self.img.shape) / np.array(self.cam.shape)
        self.cam_resized = zoom(self.cam, scale_factor, order=1)
        self.cam_resized[~self.mask] = 0 
        
        c_min, c_max = self.cam_resized.min(), self.cam_resized.max()
        if c_max > c_min:
            self.cam_resized = (self.cam_resized - c_min) / (c_max - c_min)
        
        self.update_plot(None)

    def update_plot(self, val):
        # ... (update_plot 內容與你原本的邏輯大致相同) ...
        # 唯一建議：在 title 加入目前的 Layer 顯示
        p = [self.slider0.val/100, self.slider1.val/100, self.slider2.val/100]
        sz = self.img.shape
        s = [int((sz[0]-1)*p[0]), int((sz[1]-1)*p[1]), int((sz[2]-1)*p[2])]
        
        is_correct = (int(self.info['label']) == int(self.info['prediction']))
        title_color = '#2ecc71' if is_correct else '#e74c3c'
        
        # 在標題顯示目前的層級
        self.fig.suptitle(
            f"ID: {self.info['id']} | Layer: {self.current_layer.upper()} | Pred: {self.info['prediction']} ({self.info['confidence']:.2f})\n"
            f"Result: {'CORRECT' if is_correct else 'WRONG'} | (←/→: Switch Subject)", 
            fontsize=16, color=title_color, fontweight='bold'
        )
        
        views = [
            (self.img[s[0], :, :], self.cam_resized[s[0], :, :]),
            (self.img[:, s[1], :], self.cam_resized[:, s[1], :]),
            (self.img[:, :, s[2]], self.cam_resized[:, :, s[2]])
        ]

        for i, (v_img, v_cam) in enumerate(views):
            self.axes[i].clear()
            self.axes[i].imshow(v_img.T, cmap='gray', origin='lower')
            if self.show_heatmap:
                im = self.axes[i].imshow(v_cam.T, cmap='jet', alpha=0.4, origin='lower', vmin=0, vmax=1)
                if i == 2:
                    if self.cbar is None: self.cbar = self.fig.colorbar(im, cax=self.cbar_ax)
                    else: self.cbar.update_normal(im)
                    self.cbar_ax.set_visible(True)
            else:
                if self.cbar_ax: self.cbar_ax.set_visible(False)
            self.axes[i].axis('off')
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == 'right':
            self.idx = (self.idx + 1) % len(self.subjects)
            self.load_subject()
        elif event.key == 'left':
            self.idx = (self.idx - 1) % len(self.subjects)
            self.load_subject()

if __name__ == "__main__":
    # 🔑 指向你生成的 ResNet18_3D.gradcam_multi_layer 資料夾
    viewer = GradCamViewer(r"ResNet18_3D.gradcam_multi_layer")
    plt.show()