import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import MultiModalNet 
from dataset import ChestXrayDataset 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(CURRENT_DIR, '..', 'model_loss_21_15', 'best_model.pth')
REPORT_CSV = os.path.join(CURRENT_DIR, '..', 'archive', 'indiana_reports.csv')
LABEL_CSV = os.path.join(CURRENT_DIR, '..', 'dataset_with_labels_2.csv')
IMG_DIR = os.path.join(CURRENT_DIR, '..', 'archive', 'images', 'images_normalized')

TARGET_IMG_NAME = '4_IM-2050-1001.dcm.png' 

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :]
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.text = None
    
    def forward(self, x):
        out_spec, _ = self.model(x, text_list=self.text)
        return out_spec

def main():
    dataset = ChestXrayDataset(REPORT_CSV, LABEL_CSV, IMG_DIR, split='test')
    
    target_idx = dataset.df[dataset.df['filename'].astype(str).str.strip() == TARGET_IMG_NAME].index
    if len(target_idx) == 0:
        print(f"在数据集中找不到图片 {TARGET_IMG_NAME}")
        return
    target_idx = target_idx[0]

    image_tensor, clean_text, label_specific, label_region = dataset[target_idx]

    original_model = MultiModalNet().to(DEVICE)
    original_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    original_model.eval()
    
    model = ModelWrapper(original_model)
    target_layers = [original_model.image_encoder.model.encoder.layer[-1].layernorm_before]

    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    text_input = [clean_text]

    model.text = text_input

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    true_indices = np.where(label_specific.numpy() == 1.0)[0].tolist()
    pred_indices = np.where(probs > 0.5)[0].tolist()

    true_classes = [dataset.specific_classes[i] for i in true_indices]
    pred_classes = [dataset.specific_classes[i] for i in pred_indices]

    print("\n" + "="*40)
    print("                诊断结果对比                ")
    print("="*40)
    print(f"【真实标签】:\n -> {', '.join(true_classes) if true_classes else '全阴性 (无)'}\n")
    print(f"【模型预测】:\n -> {', '.join(pred_classes) if pred_classes else '全阴性 (无)'}")
    print("="*40 + "\n")

    if len(pred_indices) == 0:
        print("模型预测全为阴性，无需生成热力图。程序结束。")
        return

    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    img_path = os.path.join(IMG_DIR, TARGET_IMG_NAME)
    img_raw = Image.open(img_path).convert("RGB").resize((224, 224))
    img_float = np.array(img_raw) / 255.0

    targets = [ClassifierOutputTarget(idx) for idx in pred_indices]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    save_path = "gradcam_result.jpg"
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    
    print(f"✅ 融合热力图已保存为: {save_path}")
    print("请打开图片查看，红色区域即为模型关注的所有病灶的综合区域！\n")

if __name__ == "__main__":
    main()