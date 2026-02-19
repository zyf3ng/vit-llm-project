import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# 引入你的模型定义
from model import MultiModalNet 
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ================= 配置 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 替换成你那个 Spec 0.2005 的最佳模型路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, '..', 'model_loss_11', 'best_model.pth')
# 随便找一张有病的图（肺炎/结节等）
IMG_PATH = os.path.join(CURRENT_DIR, '..', 'archive', 'images', 'images_normalized', '4_IM-2050-1001.dcm.png')

# ================= 辅助函数 =================
# ViT 需要把 1D 的 Token 序列 reshape 回 2D 图片
def reshape_transform(tensor, height=14, width=14):
    # 去掉 CLS token (第一个)
    result = tensor[:, 1:, :]
    # 还原形状: [B, 196, 768] -> [B, 14, 14, 768]
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    # 调整通道顺序: [B, 768, 14, 14]
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# 包装模型：GradCAM 需要模型只输出一个 Tensor
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # 我们只关心 Specific (30类) 的热力图
        # text=None 触发纯视觉模式
        out_spec, _ = self.model(x, text_list=None)
        return out_spec

# ================= 主逻辑 =================
def main():
    # 1. 加载模型
    print("正在加载模型...")
    original_model = MultiModalNet().to(DEVICE)
    original_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    original_model.eval()
    
    # 包装一下，只取 spec 输出
    model = ModelWrapper(original_model)

    # 2. 锁定目标层 (ViT 的最后一个 LayerNorm 之前)
    # 这里的路径取决于 torchvision 版本，通常是 encoder.layers[-1].ln_1
    target_layers = [original_model.image_encoder.model.encoder.layer[-1].layernorm_before]

    # 3. 准备图片
    img_raw = Image.open(IMG_PATH).convert("RGB")
    img_raw = img_raw.resize((224, 224))
    
    # 转换为 Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_raw).unsqueeze(0).to(DEVICE) # [1, 3, 224, 224]

    # 4. 初始化 GradCAM
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    # 5. 生成热力图
    # targets=None 表示自动找分数最高的那个类（比如模型认为是肺炎，就画肺炎的热力图）
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)

    # 6. 叠加显示
    grayscale_cam = grayscale_cam[0, :]
    
    # 把原图转回 float (0-1) 用于显示
    img_float = np.array(img_raw) / 255.0
    
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    # 7. 保存结果
    save_path = "gradcam_result.jpg"
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"✅ 热力图已保存为: {save_path}")
    print("请打开图片查看，红色区域即为模型关注的病灶！")

if __name__ == "__main__":
    main()