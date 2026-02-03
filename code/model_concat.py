import torch
import torch.nn as nn
from image_encoder import ImageEncoder
from text_encoder import TextEncoder

class MultiModalNet_Concat(nn.Module):
    def __init__(self, num_specific=30, num_region=5, d_model=768, dropout=0.1):
        super().__init__()

        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

        self.dropout = nn.Dropout(dropout)

        input_dim = d_model * 2 
        
        self.classifier_specific = nn.Linear(input_dim, num_specific)

        self.classifier_region = nn.Linear(input_dim, num_region)
        
    def forward(self, images, text_list):
        img_feats = self.image_encoder(images) 
        img_feats_cls = img_feats[:, 0, :]

        txt_feats, txt_mask = self.text_encoder(text_list)
        txt_feats_cls = txt_feats[:, 0, :]
        
        combined = torch.cat((img_feats_cls, txt_feats_cls), dim=1) 
        
        combined = self.dropout(combined)
        
        logits_specific = self.classifier_specific(combined)

        logits_region = self.classifier_region(combined)
        
        return logits_specific, logits_region

if __name__ == "__main__":
    model = MultiModalNet_Concat()
    
    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_txt = ["Findings: lung opacity.", "No findings."]
    
    try:
        out_spec, out_reg = model(dummy_img, dummy_txt)
        print(f"Spec Shape: {out_spec.shape}, Reg Shape: {out_reg.shape}")
        if out_spec.shape == (2, 30) and out_reg.shape == (2, 5):
            print("模型测试通过")
            
    except Exception as e:
        print(f"运行报错: {e}")