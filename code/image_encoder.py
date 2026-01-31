import torch
import torch.nn as nn
from transformers import ViTModel

class ImageEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224"):
        super().__init__()
        self.model = ViTModel.from_pretrained(model_name, add_pooling_layer=False)

    def forward(self, images):
        outputs = self.model(pixel_values=images)
        return outputs.last_hidden_state

if __name__ == "__main__":
    dummy_images = torch.randn(2, 3, 224, 224)

    encoder = ImageEncoder()
    
    feats = encoder(dummy_images)
    
    print(f"输入形状: {dummy_images.shape}")
    print(f"输出形状: {feats.shape}")
    
    if feats.shape == (2, 197, 768):
        print("测试通过！")
    else:
        print("形状不对")