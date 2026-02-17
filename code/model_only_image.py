import torch
import torch.nn as nn
from image_encoder import ImageEncoder

class MultiModalNet_OnlyImage(nn.Module):
    def __init__(self, num_specific=30, num_region=5, d_model=768, dropout=0.1):
        super().__init__()

        self.image_encoder = ImageEncoder()
        
        self.ln = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

        self.classifier_specific = nn.Linear(d_model, num_specific)
        self.classifier_region = nn.Linear(d_model, num_region)
        
    def forward(self, images, text_list=None):
        img_feats = self.image_encoder(images) 
        
        img_feats = self.ln(img_feats)

        cls_feat = img_feats[:, 0, :] 
            
        cls_feat = self.dropout(cls_feat)
        
        logits_specific = self.classifier_specific(cls_feat)
        logits_region = self.classifier_region(cls_feat)
        
        return logits_specific, logits_region

if __name__ == "__main__":
    model = MultiModalNet_OnlyImage()

    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_txt = ["some text", "text"] 
    
    try:
        out_spec, out_reg = model(dummy_img, dummy_txt)
        print(f"Spec Shape: {out_spec.shape}")
        print(f"Reg Shape:  {out_reg.shape}")
        
        if out_spec.shape == (2, 30) and out_reg.shape == (2, 5):
            print("模型测试通过")
            
    except Exception as e:
        print(f"运行报错: {e}")