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
        
    def forward(self, images, text_list=None):
        img_feats = self.image_encoder(images) 
        img_feats_cls = img_feats[:, 0, :]

        is_pure_vision = False
        if text_list is None:
            is_pure_vision = True
        elif isinstance(text_list, list) and all(t == "" for t in text_list):
            is_pure_vision = True

        if is_pure_vision:
            txt_feature = torch.zeros_like(img_feats_cls)
        else:
            txt_feats, txt_mask = self.text_encoder(text_list)
            input_mask_expanded = txt_mask.unsqueeze(-1).expand(txt_feats.size()).float()
            sum_embeddings = torch.sum(txt_feats * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            txt_feature = sum_embeddings / sum_mask
        
        combined = torch.cat((img_feats_cls, txt_feature), dim=1) 
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
        
        out_spec_vis, out_reg_vis = model(dummy_img, ["", ""])
        print(f"Vis Spec Shape: {out_spec_vis.shape}")
        print("模型测试通过")
    except Exception as e:
        print(f"运行报错: {e}")