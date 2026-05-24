import os
import sys
import json
import csv
import glob
import shutil
import torch
import numpy as np
import nibabel as nib
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import zoom
import threading
from pathlib import Path

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from Preprocess.registration                     import run_registration
from Preprocess.intensity_normalization_and_clip  import convert_to_npy
from Preprocess.back_remove                      import remove_background

# ── 常數 ──────────────────────────────────────────────────────────────────────
LAYERS         = ['layer2', 'layer3', 'layer4']
CLASS_NAMES    = ["CN", "AD"]
OUTPUT_PREPROC = os.path.join(_DEMO_DIR, "output_preprocessed")
OUTPUT_GRADCAM = os.path.join(_DEMO_DIR, "output_gradcam")

# ── 色彩系統 ──────────────────────────────────────────────────────────────────
BG_DEEP    = "#080c17"
BG_CARD    = "#0b1224"
BG_PANEL   = "#0a0f1e"
BORDER     = "#1e3a5f"
ACCENT     = "#00aaff"
ACCENT2    = "#0066cc"
TEXT_PRI   = "#ffffff"
TEXT_SEC   = "#ffffff"
TEXT_MUTED = "#ffffff"
SUCCESS    = "#00cc88"
WARNING    = "#ffaa00"
DANGER     = "#ff4466"
TEXT_BRIGHT = "#ffffff"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ──────────────────────────────────────────────────────────────────────────────
#  輔助函數
# ──────────────────────────────────────────────────────────────────────────────
def nifti_to_numpy(path):
    data = nib.load(path).get_fdata()
    return np.transpose(data, (2, 1, 0)).astype(np.float32)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_csv_info(csv_path, subject_id):
    """從 ADNI CSV 以 Image Data ID 或 Subject 欄位比對受試者"""
    try:
        with open(csv_path, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get('Image Data ID', '').strip() == subject_id.strip() or
                        row.get('Subject', '').strip() == subject_id.strip()):
                    return row
    except Exception:
        pass
    return {}

# ──────────────────────────────────────────────────────────────────────────────
#  Grad-CAM 核心
# ──────────────────────────────────────────────────────────────────────────────
def run_gradcam_single(model, img_tensor, subject_id, label, output_base, device):
    sub_folder = os.path.join(output_base, subject_id)
    os.makedirs(sub_folder, exist_ok=True)
    np.save(os.path.join(sub_folder, 'raw_mri.npy'),
            img_tensor.detach().cpu().numpy().squeeze())

    gradients  = {}
    layer_info = {}
    pred_final = prob_final = None

    def make_hook(name):
        def fn(grad): gradients[name] = grad
        return fn

    for layer_name in LAYERS:
        inp = img_tensor.detach().clone().to(device)
        inp.requires_grad_(True)
        logits, features = model(inp, return_features=True,
                                 target_layer_name=layer_name)
        handle = features.register_hook(make_hook(layer_name))
        pred   = torch.argmax(logits, dim=1).item()
        model.zero_grad()
        logits[:, pred].backward()

        grad       = gradients[layer_name]
        weights    = torch.mean(grad, dim=(2, 3, 4), keepdim=True)
        cam        = torch.relu(torch.sum(weights * features, dim=1).squeeze(0))
        cam_np     = cam.detach().cpu().numpy()
        importance = torch.mean(torch.abs(grad)).item()
        handle.remove()

        np.save(os.path.join(sub_folder, f'heatmap_{layer_name}.npy'), cam_np)
        layer_info[layer_name] = {"importance_gradient": float(importance)}
        prob_final = torch.softmax(logits, dim=1)[0, pred].item()
        pred_final = pred

    save_json({
        "id": subject_id, "label": int(label),
        "prediction": int(pred_final), "confidence": float(prob_final),
        "layers_metrics": layer_info
    }, os.path.join(sub_folder, 'info.json'))
    return sub_folder


# ══════════════════════════════════════════════════════════════════════════════
#  主應用程式
# ══════════════════════════════════════════════════════════════════════════════
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ResNet18模型")
        self.geometry("1680x960")
        self.minsize(1400, 800)
        self.configure(fg_color=BG_DEEP)

        # ── 狀態 ──────────────────────────────────────────────────────────────
        self.model               = None
        self.device              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.raw_data            = None
        self.proc_data           = None
        self.cam_data            = {}
        self.current_layer       = "layer4"
        self.show_cam            = True
        self.view_mode           = "original"
        self.cam_opacity         = 0.45
        self.cam_threshold       = 0.0
        self.slices              = [90, 108, 90]
        self.current_nii_path    = None
        self.subject_queue       = []          # 累積的影像清單
        self.current_subject_idx = 0
        self.csv_path            = None
        self.FONT_SCALE          = 1.25

        # step 0: 載入模型  step 1: 選擇影像  step 2: 前處理  step 3: Grad-CAM
        self.step_done = [False, False, False, False]

        os.makedirs(OUTPUT_PREPROC, exist_ok=True)
        os.makedirs(OUTPUT_GRADCAM, exist_ok=True)

        self._build_ui()

    def F(self, size, bold=False):
        return ctk.CTkFont("Courier New", int(size * self.FONT_SCALE),
                           "bold" if bold else "normal")

    # ──────────────────────────────────────────────────────────────────────────
    #  UI 建立
    # ──────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_center()
        self._build_right_panel()

    # ── 左側 Sidebar ──────────────────────────────────────────────────────────
    def _build_sidebar(self):
        sb = ctk.CTkScrollableFrame(self, width=270, fg_color=BG_PANEL,
                                    corner_radius=0,
                                    scrollbar_button_color=BORDER)
        sb.grid(row=0, column=0, sticky="nsew")

        fr = ctk.CTkFrame(sb, fg_color="transparent")
        fr.pack(fill="x", padx=16, pady=(22, 4))
        ctk.CTkLabel(fr, text="⬡  Resnet18",
                     font=self.F(13, True), text_color=ACCENT).pack(anchor="w")
        ctk.CTkLabel(fr, text="3D MRI  ·  Grad-CAM Analysis",
                     font=ctk.CTkFont("Courier New", 9),
                     text_color=TEXT_MUTED).pack(anchor="w")
        self._div(sb)

        # STEP 1
        self._sec(sb, "STEP 1 · MODEL")
        self.btn_model  = self._btn(sb, "⬆  載入權重 (.pth)", self._load_model, ACCENT2)
        self.lbl_model  = self._lbl(sb, "no model loaded")
        self.lbl_device = self._lbl(sb, f"device: {'CUDA ✓' if self.device.type=='cuda' else 'CPU'}")
        self._div(sb)

        # STEP 2  ── 新增「清除佇列」按鈕
        self._sec(sb, "STEP 2 · INPUT")
        self.btn_folder = self._btn(sb, "📂  新增資料夾（批次）", self._pick_folder)
        self.btn_single = self._btn(sb, "🖼  新增單一 NIfTI",    self._pick_single_nii)
        self.btn_csv    = self._btn(sb, "📋  載入 ADNI CSV",      self._pick_csv)
        self.btn_clear_queue = self._btn(sb, "🗑  清除影像佇列",  self._clear_queue, "#2a1020")
        self.lbl_input  = self._lbl(sb, "no input selected")
        self.lbl_csv    = self._lbl(sb, "no csv loaded")
        self._div(sb)

        # STEP 3
        self._sec(sb, "STEP 3 · PREPROCESS")
        self.btn_preproc = self._btn(sb, "⚙  執行前處理", self._run_preprocess, "#1a3a5c")
        self.lbl_preproc = self._lbl(sb, "—")
        self._div(sb)

        # STEP 4
        self._sec(sb, "STEP 4 · GRAD-CAM")
        self.btn_gradcam = self._btn(sb, "🧠  生成 Grad-CAM", self._run_inference, "#0d2b1e")
        self.btn_clear   = self._btn(sb, "🗑  清除結果",       self._clear_results, "#2a1020")

        ctk.CTkLabel(sb, text="TARGET LAYER",
                     font=ctk.CTkFont("Courier New", 9),
                     text_color=TEXT_MUTED).pack(anchor="w", padx=16, pady=(10, 2))
        self.layer_var = ctk.StringVar(value="layer4")
        for lyr in LAYERS:
            ctk.CTkRadioButton(
                sb, text=lyr, variable=self.layer_var, value=lyr,
                command=self._on_layer_change,
                font=self.F(13), text_color=TEXT_BRIGHT,
                radiobutton_width=14, radiobutton_height=14,
                border_width_checked=4, fg_color=ACCENT, border_color=BORDER
            ).pack(anchor="w", padx=24, pady=2)
        self._div(sb)

        self._sec(sb, "LAYER IMPORTANCE")
        self.imp_fr = ctk.CTkFrame(sb, fg_color="transparent")
        self.imp_fr.pack(fill="x", padx=12, pady=4)
        self._render_importance({})
        self._div(sb)

        self._sec(sb, "SUBJECT INFO")
        self.subj_card = ctk.CTkFrame(sb, fg_color=BG_CARD, corner_radius=10,
                                      border_width=1, border_color=BORDER)
        self.subj_card.pack(fill="x", padx=12, pady=4)
        self._render_subject_info({})
        self._div(sb)

        self._sec(sb, "OUTPUT PATHS")
        self._lbl(sb, "preproc → output_preprocessed/")
        self._lbl(sb, "gradcam → output_gradcam/")

        self._update_step_locks()

    # ── 中間主視覺區 ──────────────────────────────────────────────────────────
    def _build_center(self):
        center = ctk.CTkFrame(self, fg_color=BG_DEEP, corner_radius=0)
        center.grid(row=0, column=1, sticky="nsew")
        center.grid_rowconfigure(1, weight=1)
        center.grid_columnconfigure(0, weight=1)

        toolbar = ctk.CTkFrame(center, fg_color=BG_PANEL, height=50, corner_radius=0)
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.grid_propagate(False)

        vmode_fr = ctk.CTkFrame(toolbar, fg_color="transparent")
        vmode_fr.pack(side="left", padx=14, pady=9)
        self.view_btns = {}
        for label, key in [("Original", "original"), ("Preprocessed", "preprocessed")]:
            b = ctk.CTkButton(
                vmode_fr, text=label, width=115, height=32,
                font=self.F(13),
                fg_color=ACCENT if key == "original" else BG_CARD,
                hover_color=ACCENT2, border_width=1, border_color=BORDER,
                corner_radius=7, command=lambda k=key: self._set_view(k))
            b.pack(side="left", padx=3)
            self.view_btns[key] = b

        self.cam_btn = ctk.CTkButton(
            toolbar, text="🌡  Heatmap  ON", width=140, height=32,
            font=self.F(13), fg_color="#0d2b1e", hover_color="#1a4a2e",
            border_width=1, border_color="#1a5c3a",
            corner_radius=7, command=self._toggle_cam)
        self.cam_btn.pack(side="left", padx=6, pady=9)

        # 左右切換影像
        nav_fr = ctk.CTkFrame(toolbar, fg_color="transparent")
        nav_fr.pack(side="left", padx=14, pady=9)

        self.btn_prev = ctk.CTkButton(
            nav_fr, text="◀  Prev", width=90, height=32,
            font=self.F(13), fg_color=BG_CARD, hover_color=ACCENT2,
            border_width=1, border_color=BORDER,
            corner_radius=7, command=self._prev_subject)
        self.btn_prev.pack(side="left", padx=3)

        self.lbl_nav = ctk.CTkLabel(
            nav_fr, text="—/—", font=self.F(12), text_color=TEXT_MUTED, width=60)
        self.lbl_nav.pack(side="left", padx=4)

        self.btn_next = ctk.CTkButton(
            nav_fr, text="Next  ▶", width=90, height=32,
            font=self.F(13), fg_color=BG_CARD, hover_color=ACCENT2,
            border_width=1, border_color=BORDER,
            corner_radius=7, command=self._next_subject)
        self.btn_next.pack(side="left", padx=3)

        self.lbl_coord = ctk.CTkLabel(
            toolbar, text="X:—  Y:—  Z:—", font=self.F(13), text_color=TEXT_MUTED)
        self.lbl_coord.pack(side="right", padx=16)

        views_fr = ctk.CTkFrame(center, fg_color=BG_DEEP, corner_radius=0)
        views_fr.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        views_fr.grid_columnconfigure((0, 1, 2), weight=1)
        views_fr.grid_rowconfigure(0, weight=1)

        self.panels = []
        self.view_titles = [ "Sagittal  [X]", "Coronal  [Y]","Axial  [Z]"]
        for col in range(3):
            self.panels.append(self._build_view_panel(views_fr, self.view_titles[col], col))

        ctrl = ctk.CTkFrame(center, fg_color=BG_PANEL, height=110, corner_radius=0)
        ctrl.grid(row=2, column=0, sticky="ew")
        ctrl.grid_propagate(False)
        ctrl.grid_columnconfigure((0, 1, 2), weight=1)

        self._bottom_slider(ctrl, 0, "SLICE",     0, 255, 90,  self._on_slice_all)
        self._bottom_slider(ctrl, 1, "OPACITY",   0, 100, 45,  self._on_opacity,
                            fmt=lambda v: f"{int(v)}%")
        self._bottom_slider(ctrl, 2, "THRESHOLD", 0, 100, 0,   self._on_threshold,
                            fmt=lambda v: f"{int(v)}%")

    def _build_view_panel(self, parent, title, col):
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=12,
                            border_width=1, border_color=BORDER)
        card.grid(row=0, column=col, sticky="nsew", padx=5, pady=5)
        card.grid_rowconfigure(1, weight=1)
        card.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(card, fg_color="transparent", height=30)
        hdr.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 0))
        hdr.grid_propagate(False)
        title_lbl = ctk.CTkLabel(hdr, text=title,
                                 font=ctk.CTkFont("Courier New", 11, "bold"),
                                 text_color=ACCENT)
        title_lbl.pack(side="left")

        fig, ax = plt.subplots(1, 1, facecolor=BG_CARD)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_facecolor("#050810")
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, master=card)
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        canvas.get_tk_widget().configure(bg=BG_CARD, highlightthickness=0)

        sl_fr = ctk.CTkFrame(card, fg_color="transparent")
        sl_fr.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))
        sl_fr.grid_columnconfigure(1, weight=1)

        idx_lbl = ctk.CTkLabel(sl_fr, text="0", font=self.F(11),
                               text_color=TEXT_BRIGHT, width=30)
        idx_lbl.grid(row=0, column=0, padx=(0, 4))

        slider = ctk.CTkSlider(sl_fr, from_=0, to=255, number_of_steps=255,
                               height=14, button_color=ACCENT,
                               button_hover_color=ACCENT2,
                               progress_color=ACCENT2, fg_color=BORDER)
        slider.grid(row=0, column=1, sticky="ew")
        slider.set(self.slices[col])

        rst = ctk.CTkButton(sl_fr, text="⟳", width=28, height=22,
                            font=self.F(13), fg_color=BG_PANEL, hover_color=BORDER,
                            corner_radius=6,
                            command=lambda c=col: self._reset_slice(c))
        rst.grid(row=0, column=2, padx=(4, 0))

        slider.configure(command=lambda v, c=col, l=idx_lbl: self._on_single_slice(v, c, l))

        return {"fig": fig, "ax": ax, "canvas": canvas,
                "slider": slider, "idx_lbl": idx_lbl, "title_lbl": title_lbl}

    def _bottom_slider(self, parent, col, label, lo, hi, init, cb, fmt=None):
        fr = ctk.CTkFrame(parent, fg_color="transparent")
        fr.grid(row=0, column=col, sticky="ew", padx=22, pady=14)
        fr.grid_columnconfigure(0, weight=1)
        _fmt = fmt or (lambda v: str(int(v)))
        ctk.CTkLabel(fr, text=label,
                     font=ctk.CTkFont("Courier New", 10, "bold"),
                     text_color=TEXT_MUTED).grid(row=0, column=0, sticky="w")
        val_lbl = ctk.CTkLabel(fr, text=_fmt(init),
                               font=ctk.CTkFont("Courier New", 12, "bold"),
                               text_color=ACCENT)
        val_lbl.grid(row=0, column=1, sticky="e")
        sl = ctk.CTkSlider(fr, from_=lo, to=hi, number_of_steps=int(hi - lo),
                           height=16, button_color=ACCENT,
                           button_hover_color=ACCENT2,
                           progress_color=ACCENT2, fg_color=BORDER)
        sl.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        sl.set(init)
        sl.configure(command=lambda v, l=val_lbl, f=_fmt: (l.configure(text=f(v)), cb(v)))

    # ── 右側資訊面板 ──────────────────────────────────────────────────────────
    def _build_right_panel(self):
        rp = ctk.CTkFrame(self, width=250, fg_color=BG_PANEL, corner_radius=0)
        rp.grid(row=0, column=2, sticky="nsew")
        rp.grid_propagate(False)

        self._sec(rp, "PREDICTION")
        pred_card = ctk.CTkFrame(rp, fg_color=BG_CARD, corner_radius=10,
                                 border_width=1, border_color=BORDER)
        pred_card.pack(fill="x", padx=12, pady=6)
        self.lbl_pred = ctk.CTkLabel(pred_card, text="—",
                                     font=ctk.CTkFont("Courier New", 26, "bold"),
                                     text_color=ACCENT)
        self.lbl_pred.pack(pady=(16, 2))
        self.lbl_conf = ctk.CTkLabel(pred_card, text="confidence: —",
                                     font=self.F(13), text_color=TEXT_BRIGHT)
        self.lbl_conf.pack(pady=(0, 16))

        self._div(rp)
        self._sec(rp, "LOG")
        self.log_box = ctk.CTkTextbox(
            rp, height=300, fg_color=BG_CARD, font=self.F(11), text_color=TEXT_BRIGHT,
            border_width=1, border_color=BORDER, corner_radius=8,
            scrollbar_button_color=BORDER)
        self.log_box.pack(fill="x", padx=12, pady=6)
        self._log("System ready.")
        self._log(f"Device: {self.device}")

        self._div(rp)
        self._sec(rp, "QUEUE")
        self.queue_box = ctk.CTkTextbox(
            rp, height=160, fg_color=BG_CARD, font=self.F(11), text_color=TEXT_MUTED,
            border_width=1, border_color=BORDER, corner_radius=8,
            scrollbar_button_color=BORDER)
        self.queue_box.pack(fill="x", padx=12, pady=6)

    # ──────────────────────────────────────────────────────────────────────────
    #  UI 小工具
    # ──────────────────────────────────────────────────────────────────────────
    def _sec(self, p, text):
        fr = ctk.CTkFrame(p, fg_color="transparent")
        fr.pack(fill="x", padx=12, pady=(12, 2))
        ctk.CTkLabel(fr, text=text,
                     font=ctk.CTkFont("Courier New", 10, "bold"),
                     text_color=TEXT_MUTED).pack(side="left")
        ctk.CTkFrame(fr, fg_color=BORDER, height=1).pack(
            side="left", fill="x", expand=True, padx=(8, 0), pady=6)

    def _div(self, p):
        ctk.CTkFrame(p, fg_color=BORDER, height=1).pack(fill="x", padx=12, pady=6)

    def _btn(self, p, text, cmd, color=BG_CARD):
        b = ctk.CTkButton(p, text=text, height=34, font=self.F(13),
                          fg_color=color, hover_color=ACCENT2,
                          border_width=1, border_color=BORDER,
                          corner_radius=8, anchor="w", command=cmd)
        b.pack(fill="x", padx=12, pady=3)
        return b

    def _lbl(self, p, text):
        l = ctk.CTkLabel(p, text=text, font=self.F(11),
                         text_color=TEXT_MUTED, anchor="w", wraplength=220)
        l.pack(anchor="w", padx=16, pady=1)
        return l

    def _log(self, msg):
        try:
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.update_idletasks()
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────────
    #  步驟鎖定
    # ──────────────────────────────────────────────────────────────────────────
    def _update_step_locks(self):
        self.btn_model.configure(state="normal")

        s2_ok  = self.step_done[0]
        state2 = "normal" if s2_ok else "disabled"
        self.btn_folder.configure(state=state2)
        self.btn_single.configure(state=state2)
        self.btn_csv.configure(state=state2)
        self.btn_clear_queue.configure(state=state2)

        s3_ok = self.step_done[0] and self.step_done[1]
        self.btn_preproc.configure(state="normal" if s3_ok else "disabled")

        s4_ok = self.step_done[0] and self.step_done[1] and self.step_done[2]
        self.btn_gradcam.configure(state="normal" if s4_ok else "disabled")

    # ──────────────────────────────────────────────────────────────────────────
    #  動態元件渲染
    # ──────────────────────────────────────────────────────────────────────────
    def _render_importance(self, metrics: dict):
        for w in self.imp_fr.winfo_children():
            w.destroy()
        if not metrics:
            ctk.CTkLabel(self.imp_fr, text="no data",
                         font=self.F(11), text_color=TEXT_MUTED).pack(anchor="w")
            return
        vals = {k: v.get("importance_gradient", 0) for k, v in metrics.items()}
        mx   = max(vals.values()) or 1
        for layer, val in vals.items():
            row = ctk.CTkFrame(self.imp_fr, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=layer, width=58, font=self.F(11),
                         text_color=TEXT_BRIGHT).pack(side="left")
            bg = ctk.CTkFrame(row, fg_color=BORDER, height=8, corner_radius=4)
            bg.pack(side="left", fill="x", expand=True, padx=4)
            w = max(4, int(140 * val / mx))
            ctk.CTkFrame(bg, fg_color=ACCENT, height=8,
                         corner_radius=4, width=w).place(x=0, y=0)
            ctk.CTkLabel(row, text=f"{val:.4f}",
                         font=ctk.CTkFont("Courier New", 9),
                         text_color=TEXT_MUTED, width=58).pack(side="left")

    def _render_subject_info(self, info: dict):
        for w in self.subj_card.winfo_children():
            w.destroy()
        if not info:
            ctk.CTkLabel(self.subj_card, text="no subject selected",
                         font=self.F(11), text_color=TEXT_MUTED).pack(pady=10)
            return
        fields = [
            ("ID",       info.get("Image Data ID", info.get("Subject", "—"))),
            ("Group",    info.get("Group",    "—")),
            ("Sex",      info.get("Sex",      "—")),
            ("Age",      info.get("Age",      "—")),
            ("Visit",    info.get("Visit",    "—")),
            ("Acq Date", info.get("Acq Date", "—")),
        ]
        for key, val in fields:
            row = ctk.CTkFrame(self.subj_card, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=2)
            ctk.CTkLabel(row, text=key, width=64, font=self.F(11),
                         text_color=TEXT_MUTED, anchor="w").pack(side="left")
            color = (SUCCESS if val == "CN" else DANGER if val == "AD" else TEXT_PRI)
            ctk.CTkLabel(row, text=str(val),
                         font=ctk.CTkFont("Courier New", 10, "bold"),
                         text_color=color, anchor="w").pack(side="left")

    def _reset_prediction_ui(self):
        """清空預測結果、CAM、Layer Importance（不動受試者資料）"""
        self.cam_data  = {}
        self.proc_data = None
        self.lbl_pred.configure(text="—", text_color=ACCENT)
        self.lbl_conf.configure(text="confidence: —")
        self._render_importance({})

    # ──────────────────────────────────────────────────────────────────────────
    #  事件：輸入
    # ──────────────────────────────────────────────────────────────────────────
    def _load_model(self):
        path = filedialog.askopenfilename(filetypes=[("Weights", "*.pth")])
        if not path:
            return
        try:
            from models.model import ResNet18_3D
            self.model = ResNet18_3D(fil_num=20, drop_rate=0.2).to(self.device)
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt.get('model_state_dict', ckpt))
            self.model.eval()
            name = os.path.basename(path)
            self.lbl_model.configure(text=name, text_color=SUCCESS)
            self._log(f"Model: {name}")
            self.step_done[0] = True
            self._update_step_locks()
        except Exception as e:
            messagebox.showerror("Model Error", str(e))

    # ── 累積新增影像（不清除舊佇列）─────────────────────────────────────────
    def _pick_folder(self):
        folder = filedialog.askdirectory(title="選擇含 .nii 的資料夾")
        if not folder:
            return
        files = sorted(
            glob.glob(os.path.join(folder, "**/*.nii"),    recursive=True) +
            glob.glob(os.path.join(folder, "**/*.nii.gz"), recursive=True)
        )
        if not files:
            self._log(f"Folder: {os.path.basename(folder)} — 沒有找到 NIfTI 檔案")
            return

        # 去除重複，已在佇列中的不再加入
        existing = set(self.subject_queue)
        added = [f for f in files if f not in existing]
        if not added:
            self._log(f"Folder: {os.path.basename(folder)} — 所有檔案已在佇列中")
            return

        first_new_idx = len(self.subject_queue)          # 記住新增起點
        self.subject_queue.extend(added)
        self._log(f"Folder: {os.path.basename(folder)}  新增 {len(added)} 筆，共 {len(self.subject_queue)} 筆")
        self.current_subject_idx = first_new_idx          # 跳到剛新增的第一筆
        self._switch_to_subject(self.current_subject_idx)
        self.lbl_input.configure(
            text=f"{len(self.subject_queue)} 筆影像已載入", text_color=TEXT_BRIGHT)
        self.step_done[1] = True
        self._update_step_locks()

    def _pick_single_nii(self):
        path = filedialog.askopenfilename(filetypes=[("NIfTI", "*.nii *.nii.gz")])
        if not path:
            return
        if path in self.subject_queue:
            # 已在佇列中 → 直接跳到該影像
            idx = self.subject_queue.index(path)
            self._log(f"已在佇列中，切換至 [{idx+1}] {os.path.basename(path)}")
            self.current_subject_idx = idx
            self._switch_to_subject(idx)
            return
        self.subject_queue.append(path)
        self._log(f"新增：{os.path.basename(path)}  共 {len(self.subject_queue)} 筆")
        self._after_add_to_queue()

    def _after_add_to_queue(self):
        """每次新增影像後的共用處理：跳到最新加入的第一張並更新 UI"""
        # 跳到剛加入的第一筆（即整個佇列最新的末端起點）
        # 策略：若目前是空的剛新增，直接跳 0；否則跳到新增第一筆
        # 簡單方式：跳到佇列最後一筆
        # （使用者可以透過 Prev 回去看舊的）
        self.current_subject_idx = len(self.subject_queue) - 1
        self._switch_to_subject(self.current_subject_idx)

        total = len(self.subject_queue)
        self.lbl_input.configure(
            text=f"{total} 筆影像已載入", text_color=TEXT_BRIGHT)
        self.step_done[1] = True
        self._update_step_locks()

    def _clear_queue(self):
        """清除整個影像佇列"""
        if not self.subject_queue:
            return
        if not messagebox.askyesno("清除佇列", f"確定要清除全部 {len(self.subject_queue)} 筆影像？"):
            return
        self.subject_queue       = []
        self.current_subject_idx = 0
        self.current_nii_path    = None
        self.raw_data            = None
        self.proc_data           = None
        self.cam_data            = {}
        self.view_mode           = "original"
        self.step_done[1]        = False
        self.step_done[2]        = False
        self.step_done[3]        = False
        self.lbl_input.configure(text="no input selected", text_color=TEXT_MUTED)
        self.lbl_preproc.configure(text="—", text_color=TEXT_MUTED)
        self._reset_prediction_ui()
        self._render_subject_info({})
        self._update_queue_box()
        self._update_nav_label()
        self._update_view_btns()
        self._refresh_views()
        self._update_step_locks()
        self._log("佇列已清除。")

    def _pick_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return
        self.csv_path = path
        self.lbl_csv.configure(text=os.path.basename(path), text_color=TEXT_BRIGHT)
        self._log(f"CSV: {os.path.basename(path)}")

    # ──────────────────────────────────────────────────────────────────────────
    #  佇列 UI 更新
    # ──────────────────────────────────────────────────────────────────────────
    def _update_queue_box(self):
        self.queue_box.delete("1.0", "end")
        for i, f in enumerate(self.subject_queue):
            prefix = "▶ " if i == self.current_subject_idx else "  "
            self.queue_box.insert("end", f"{prefix}[{i+1:02d}] {os.path.basename(f)}\n")

    def _update_nav_label(self):
        total = len(self.subject_queue)
        idx   = self.current_subject_idx
        if total == 0:
            self.lbl_nav.configure(text="—/—")
            self.btn_prev.configure(state="disabled", fg_color=BG_CARD)
            self.btn_next.configure(state="disabled", fg_color=BG_CARD)
        else:
            self.lbl_nav.configure(text=f"{idx + 1}/{total}")
            self.btn_prev.configure(
                state="normal"   if idx > 0          else "disabled",
                fg_color=ACCENT2 if idx > 0          else BG_CARD)
            self.btn_next.configure(
                state="normal"   if idx < total - 1  else "disabled",
                fg_color=ACCENT2 if idx < total - 1  else BG_CARD)

    # ──────────────────────────────────────────────────────────────────────────
    #  NIfTI 預覽載入
    # ──────────────────────────────────────────────────────────────────────────
    def _load_nii_preview(self, path, reset_state=True):
        """
        載入 NIfTI 為 raw_data 並重設 slider。
        reset_state=True  → 清空 proc_data/cam_data，切到 original（首次載入用）
        reset_state=False → 保留呼叫方已設好的狀態（_switch_to_subject 用）
        """
        try:
            self.raw_data = nifti_to_numpy(path)
            if reset_state:
                self.proc_data = None
                self.cam_data  = {}
                self.view_mode = "original"
            shape = self.raw_data.shape
            for i, p in enumerate(self.panels):
                mx = shape[i] - 1
                p["slider"].configure(to=mx)
                self.slices[i] = shape[i] // 2
                p["slider"].set(self.slices[i])
                p["idx_lbl"].configure(text=str(self.slices[i]))
            self._update_view_btns()
            self._refresh_views()
        except Exception as e:
            self._log(f"Preview error: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    #  左右切換影像
    # ──────────────────────────────────────────────────────────────────────────
    def _prev_subject(self):
        if self.subject_queue and self.current_subject_idx > 0:
            self.current_subject_idx -= 1
            self._switch_to_subject(self.current_subject_idx)

    def _next_subject(self):
        if self.subject_queue and self.current_subject_idx < len(self.subject_queue) - 1:
            self.current_subject_idx += 1
            self._switch_to_subject(self.current_subject_idx)

    def _switch_to_subject(self, idx):
        path = self.subject_queue[idx]
        self.current_nii_path = path
        self._log(f"Switch → [{idx+1}/{len(self.subject_queue)}] {os.path.basename(path)}")
        self._update_nav_label()
        self._update_queue_box()

        sid = Path(path).stem.replace(".nii", "")

        # 1. 清空當前結果
        self.cam_data  = {}
        self.proc_data = None
        self.lbl_pred.configure(text="—", text_color=ACCENT)
        self.lbl_conf.configure(text="confidence: —")
        self._render_importance({})

        # 2. 更新 CSV 受試者資料
        csv_row = load_csv_info(self.csv_path, sid) if self.csv_path else {}
        self._render_subject_info(csv_row)

        # 3. 嘗試載入前處理結果
        npy_path = os.path.join(OUTPUT_PREPROC, sid + ".npy")
        if os.path.exists(npy_path):
            try:
                self.proc_data = np.load(npy_path).astype(np.float32)
                self._log(f"  ↳ 已載入前處理結果")
            except Exception as e:
                self._log(f"  ↳ 前處理結果載入失敗: {e}")

        # 4. 嘗試載入 Grad-CAM 結果
        sub_folder = os.path.join(OUTPUT_GRADCAM, sid)
        info_path  = os.path.join(sub_folder, 'info.json')
        has_cam    = False
        if os.path.exists(info_path):
            try:
                with open(info_path, encoding='utf-8') as f:
                    info = json.load(f)
                ref = self.proc_data if self.proc_data is not None else nifti_to_numpy(path)
                for layer in LAYERS:
                    fp = os.path.join(sub_folder, f'heatmap_{layer}.npy')
                    if os.path.exists(fp):
                        raw = np.load(fp)
                        if raw.max() > raw.min():
                            scale = np.array(ref.shape) / np.array(raw.shape)
                            cam_r = zoom(raw, scale, order=1)
                            cam_r = (cam_r - cam_r.min()) / (cam_r.max() - cam_r.min())
                            self.cam_data[layer] = cam_r
                pred_name = (CLASS_NAMES[info['prediction']]
                             if 0 <= info['prediction'] < len(CLASS_NAMES) else "?")
                color = DANGER if pred_name == "AD" else SUCCESS
                self.lbl_pred.configure(text=pred_name, text_color=color)
                self.lbl_conf.configure(text=f"confidence: {info['confidence']:.2%}")
                self._render_importance(info.get('layers_metrics', {}))
                has_cam = True
                self._log(f"  ↳ 已載入 Grad-CAM 結果")
            except Exception as e:
                self._log(f"  ↳ Grad-CAM 結果載入失敗: {e}")

        # 5. 決定 view_mode
        if has_cam and self.proc_data is not None:
            self.view_mode = "preprocessed"
            self.show_cam  = True
            self.cam_btn.configure(text="🌡  Heatmap  ON", fg_color="#0d2b1e")
        elif self.proc_data is not None:
            self.view_mode = "preprocessed"
        else:
            self.view_mode = "original"

        # 6. 載入 raw NIfTI（保留上面設好的狀態）
        self._load_nii_preview(path, reset_state=False)

    # ──────────────────────────────────────────────────────────────────────────
    #  前處理
    # ──────────────────────────────────────────────────────────────────────────
    def _run_preprocess(self):
        if not self.subject_queue:
            messagebox.showwarning("警告", "請先選擇影像")
            return

        def worker():
            queue_snapshot = list(self.subject_queue)
            total = len(queue_snapshot)
            for i, nii_path in enumerate(queue_snapshot):
                sid = Path(nii_path).stem.replace(".nii", "")
                self._log(f"[{i+1}/{total}] preproc: {sid}")
                self.lbl_preproc.configure(text=f"{i+1}/{total}: {sid}", text_color=WARNING)
                try:
                    reg   = run_registration(nii_path)
                    npy   = convert_to_npy(reg)
                    final = remove_background(npy)
                    dst   = os.path.join(OUTPUT_PREPROC, sid + ".npy")
                    shutil.copy2(final, dst)
                    self._log(f"  ✓ {sid}.npy")

                    if nii_path == self.current_nii_path:
                        loaded = np.load(dst).astype(np.float32)
                        def _upd(data=loaded):
                            self.proc_data = data
                            self.view_mode = "preprocessed"
                            self._update_view_btns()
                            self._refresh_views()
                            self._log("  ↳ 已切換至前處理結果視圖")
                        self.after(0, _upd)
                except Exception as e:
                    self._log(f"  ✗ {sid}: {e}")

            def _done():
                self.lbl_preproc.configure(text=f"done ({total})", text_color=SUCCESS)
                self._log("Preprocess complete.")
                self.step_done[2] = True
                self._update_step_locks()
            self.after(0, _done)

        threading.Thread(target=worker, daemon=True).start()

    # ──────────────────────────────────────────────────────────────────────────
    #  推論 + Grad-CAM
    # ──────────────────────────────────────────────────────────────────────────
    def _run_inference(self):
        if self.model is None:
            messagebox.showwarning("警告", "請先載入模型"); return
        if not self.subject_queue:
            messagebox.showwarning("警告", "請先選擇影像"); return

        def worker():
            queue_snapshot = list(self.subject_queue)
            total = len(queue_snapshot)
            for i, nii_path in enumerate(queue_snapshot):
                sid = Path(nii_path).stem.replace(".nii", "")
                self._log(f"[{i+1}/{total}] inference: {sid}")

                npy_path = os.path.join(OUTPUT_PREPROC, sid + ".npy")
                if os.path.exists(npy_path):
                    data = np.load(npy_path).astype(np.float32)
                else:
                    data = nifti_to_numpy(nii_path)
                    self._log("  (no preproc → using raw)")

                csv_row = {}
                if self.csv_path:
                    csv_row = load_csv_info(self.csv_path, sid) or {}
                label = (0 if csv_row.get("Group", "").strip() == "CN" else
                         1 if csv_row.get("Group", "").strip() == "AD" else -1)

                try:
                    img_t = (torch.from_numpy(data)
                             .unsqueeze(0).unsqueeze(0).float().to(self.device))
                    sub_folder = run_gradcam_single(
                        self.model, img_t, sid, label, OUTPUT_GRADCAM, self.device)

                    with open(os.path.join(sub_folder, 'info.json'), encoding='utf-8') as f:
                        info = json.load(f)

                    pred_name = (CLASS_NAMES[info['prediction']]
                                 if 0 <= info['prediction'] < len(CLASS_NAMES) else "?")
                    self._log(f"  pred={pred_name}  conf={info['confidence']:.2%}")

                    if nii_path == self.current_nii_path:
                        def _upd(sf=sub_folder, cr=csv_row, nfo=info, d=data):
                            self.proc_data = d
                            self._display_result(sf, cr, nfo)
                        self.after(0, _upd)

                except Exception as e:
                    self._log(f"  ✗ {sid}: {e}")

            def _done():
                self._log("Inference complete.")
                self.step_done[3] = True
                self._update_step_locks()
            self.after(0, _done)

        threading.Thread(target=worker, daemon=True).start()

    def _display_result(self, sub_folder, csv_row, info):
        data = self.proc_data if self.proc_data is not None else self.raw_data
        self.cam_data = {}
        for layer in LAYERS:
            fp = os.path.join(sub_folder, f'heatmap_{layer}.npy')
            if os.path.exists(fp) and data is not None:
                raw = np.load(fp)
                if raw.max() > raw.min():
                    scale = np.array(data.shape) / np.array(raw.shape)
                    cam_r = zoom(raw, scale, order=1)
                    cam_r = (cam_r - cam_r.min()) / (cam_r.max() - cam_r.min())
                    self.cam_data[layer] = cam_r

        pred_name = (CLASS_NAMES[info['prediction']]
                     if 0 <= info['prediction'] < len(CLASS_NAMES) else "?")
        color = DANGER if pred_name == "AD" else SUCCESS
        self.lbl_pred.configure(text=pred_name, text_color=color)
        self.lbl_conf.configure(text=f"confidence: {info['confidence']:.2%}")
        self._render_importance(info.get('layers_metrics', {}))
        self._render_subject_info(csv_row)

        self.view_mode = "preprocessed"
        self._update_view_btns()
        self.show_cam = True
        self.cam_btn.configure(text="🌡  Heatmap  ON", fg_color="#0d2b1e")
        self._refresh_views()

    def _clear_results(self):
        self.cam_data  = {}
        self.proc_data = None
        self.lbl_pred.configure(text="—", text_color=ACCENT)
        self.lbl_conf.configure(text="confidence: —")
        self._render_importance({})
        self._render_subject_info({})
        self._refresh_views()
        self._log("Results cleared.")

    # ──────────────────────────────────────────────────────────────────────────
    #  視圖控制
    # ──────────────────────────────────────────────────────────────────────────
    def _set_view(self, mode):
        if mode == "preprocessed" and self.proc_data is None:
            self._log("尚無前處理結果，請先執行前處理。")
            return
        self.view_mode = mode
        self._update_view_btns()
        self._refresh_views()

    def _update_view_btns(self):
        for k, b in self.view_btns.items():
            b.configure(fg_color=ACCENT if k == self.view_mode else BG_CARD)

    def _toggle_cam(self):
        self.show_cam = not self.show_cam
        self.cam_btn.configure(
            text=f"🌡  Heatmap  {'ON' if self.show_cam else 'OFF'}",
            fg_color="#0d2b1e" if self.show_cam else BG_CARD)
        self._refresh_views()

    def _on_layer_change(self):
        self.current_layer = self.layer_var.get()
        self._refresh_views()

    def _on_single_slice(self, val, col, lbl):
        self.slices[col] = int(float(val))
        lbl.configure(text=str(self.slices[col]))
        self._update_coord()
        self._refresh_views()

    def _on_slice_all(self, val):
        v    = int(float(val))
        data = self._active_data()
        for i, p in enumerate(self.panels):
            sv = min(v, data.shape[i] - 1) if data is not None else v
            self.slices[i] = sv
            p["slider"].set(sv)
            p["idx_lbl"].configure(text=str(sv))
        self._update_coord()
        self._refresh_views()

    def _on_opacity(self, val):
        self.cam_opacity = float(val) / 100
        self._refresh_views()

    def _on_threshold(self, val):
        self.cam_threshold = float(val) / 100
        self._refresh_views()

    def _reset_slice(self, col):
        data = self._active_data()
        mid  = data.shape[col] // 2 if data is not None else 90
        self.slices[col] = mid
        self.panels[col]["slider"].set(mid)
        self.panels[col]["idx_lbl"].configure(text=str(mid))
        self._update_coord()
        self._refresh_views()

    def _update_coord(self):
        self.lbl_coord.configure(
            text=f"X:{self.slices[1]:3d}  Y:{self.slices[2]:3d}  Z:{self.slices[0]:3d}")

    def _active_data(self):
        if self.view_mode == "preprocessed" and self.proc_data is not None:
            return self.proc_data
        return self.raw_data

    # ──────────────────────────────────────────────────────────────────────────
    #  畫面渲染
    # ──────────────────────────────────────────────────────────────────────────
    def _refresh_views(self):
        data = self._active_data()
        cam  = self.cam_data.get(self.current_layer)
        s    = self.slices
        show_titles = (self.view_mode == "preprocessed")

        for i, p in enumerate(self.panels):
            p["title_lbl"].configure(text=self.view_titles[i] if show_titles else "")
            ax = p["ax"]
            ax.clear()
            ax.set_facecolor("#050810")
            ax.axis('off')

            if data is not None:
                sz = data.shape
                s0 = min(s[0], sz[0]-1)
                s1 = min(s[1], sz[1]-1)
                s2 = min(s[2], sz[2]-1)
                slices_2d = [data[s0,:,:], data[:,s1,:], data[:,:,s2]]
                ax.imshow(slices_2d[i].T, cmap='gray', origin='lower', aspect='equal')
                ax.set_aspect(data.shape[i] / data.shape[(i+1)%3])

                if cam is not None and self.show_cam:
                    cam_2d = [cam[s0,:,:], cam[:,s1,:], cam[:,:,s2]]
                    masked = np.ma.masked_where(cam_2d[i].T < self.cam_threshold, cam_2d[i].T)
                    ax.imshow(masked, cmap='jet', alpha=self.cam_opacity,
                              origin='lower', aspect='equal', vmin=0, vmax=1)
                    ax.set_aspect(data.shape[i] / data.shape[(i+1)%3])
            else:
                ax.text(0.5, 0.5, "no image", ha='center', va='center',
                        color=TEXT_MUTED, fontsize=9,
                        fontfamily='monospace', transform=ax.transAxes)

            p["canvas"].draw_idle()

        self._update_coord()


# ── 進入點 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()