import torch
import torch.nn as nn
from text_encoder import TextEncoder

class MultiModalNet_OnlyText(nn.Module):
    def __init__(self, num_specific=30, num_region=5, d_model=768, dropout=0.1):
        super().__init__()

        self.text_encoder = TextEncoder()
        
        self.dropout = nn.Dropout(dropout)

        self.classifier_specific = nn.Linear(d_model, num_specific)

        self.classifier_region = nn.Linear(d_model, num_region)
        
    def forward(self, images, text_list):

        txt_feats, txt_mask = self.text_encoder(text_list)
        
        input_mask_expanded = txt_mask.unsqueeze(-1).expand(txt_feats.size()).float()

        sum_embeddings = torch.sum(txt_feats * input_mask_expanded, 1)

        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        feature = sum_embeddings / sum_mask
        
        features = self.dropout(feature)
        
        logits_specific = self.classifier_specific(features)

        logits_region = self.classifier_region(features)
        
        return logits_specific, logits_region
    
if __name__ == "__main__":
    model = MultiModalNet_OnlyText()
    
    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_txt = ["Lung opacity found.", "No findings."]
    
    try:
        out_spec, out_reg = model(dummy_img, dummy_txt)
        
        print(f"输入图片: {dummy_img.shape}")
        print(f"Specific Output: {out_spec.shape}")
        print(f"Region Output:  {out_reg.shape}")
        
        if out_spec.shape == (2, 30) and out_reg.shape == (2, 5):
            print("模型测试通过")
        else:
            print("输出形状不对")
            
    except Exception as e:
        print(f"运行报错: {e}")