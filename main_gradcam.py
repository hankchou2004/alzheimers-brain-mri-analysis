import os
import torch
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.model import ResNet18_3D # 確保引用正確
from data.dataloader import Data

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)



def save_gradient(grad):
    global gradients
    gradients = grad

def main():
    exp_idx = 4
    model_name = "ResNet18_3D"
    checkpoint_path = r"C:\Users\User\Desktop\demo\checkpoint_dir\final\ResNet18_3D_exp0\ResNet18_3D_best_metric_31.pth"
    output_base = f"./{model_name}.gradcam_multi_layer"
    os.makedirs(output_base, exist_ok=True)

    model = ResNet18_3D(fil_num=20, drop_rate=0.2).cuda()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval() # 記得設為 eval 模式

    test_dataset = Data(data_dir=r"C:/Users/User/Desktop/demo/full/", exp_idx=exp_idx, stage='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    layers_to_monitor = ['layer2', 'layer3', 'layer4']
    global gradients

    print("🚀 開始分層生成 Grad-CAM...")
    for i, (img, label) in enumerate(tqdm(test_loader)):
        img = img.cuda()
        subject_id = test_dataset.data_list[i]
        sub_folder = os.path.join(output_base, subject_id)
        os.makedirs(sub_folder, exist_ok=True)
        
        # 儲存原始 MRI (只需存一次)
        np.save(os.path.join(sub_folder, 'raw_mri.npy'), img.detach().cpu().numpy().squeeze())

        layer_info = {}
        for layer_name in layers_to_monitor:
            img.requires_grad = True
            
            # 指定層輸出
            logits, features = model(img, return_features=True, target_layer_name=layer_name)
            
            h = features.register_hook(save_gradient)
            pred = torch.argmax(logits, dim=1).item()
            score = logits[:, pred]
            
            model.zero_grad()
            score.backward()

            # Grad-CAM 計算
            weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)
            cam = torch.sum(weights * features, dim=1).squeeze(0)
            cam = torch.relu(cam).detach().cpu().numpy()
            
            # 重要性指標：計算該層梯度的平均強度 (Mean Gradient Magnitude)
            importance_score = torch.mean(torch.abs(gradients)).item()
            
            h.remove()
            
            # 儲存該層的 heatmap
            np.save(os.path.join(sub_folder, f'heatmap_{layer_name}.npy'), cam)
            layer_info[layer_name] = {"importance_gradient": importance_score}

        # ... 前面 backward 邏輯不變 ...
            
            # 🔑 在循環外或內計算一次信心度
            prob = torch.softmax(logits, dim=1)[0, pred].item()

        # 修正後的 save_json
        save_json({
            "id": subject_id,
            "label": int(label.item()),
            "prediction": pred,
            "confidence": float(prob),  # 🔑 補上這行
            "layers_metrics": layer_info
        }, os.path.join(sub_folder, 'info.json'))
if __name__ == "__main__":
    main()
