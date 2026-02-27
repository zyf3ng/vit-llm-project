import torch
import torch.nn as nn
from image_encoder import ImageEncoder
from text_encoder import TextEncoder

class MultiModalNet_NoRegion(nn.Module):
    def __init__(self, num_specific=30, d_model=768, num_heads=8, dropout=0.1):
        super().__init__()

        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim = d_model, 
            num_heads = num_heads, 
            dropout = dropout, 
            batch_first = True
        )

        self.ln = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.classifier_specific = nn.Linear(d_model, num_specific)
        
    def forward(self, images, text_list=None):
        img_feats = self.image_encoder(images)
        
        # --- 核心对齐：纯视觉特判逻辑 ---
        is_pure_vision = False
        if text_list is None:
            is_pure_vision = True
        elif isinstance(text_list, list) and all(t == "" for t in text_list):
            is_pure_vision = True
            
        if is_pure_vision:
            # 剥夺文本时，Cross-Attention 输出强行清零
            attn_output = torch.zeros_like(img_feats)
        else:
            txt_feats, txt_mask = self.text_encoder(text_list)
            padding_mask = (txt_mask == 0).bool()
            attn_output, _ = self.cross_attention(
                query = img_feats,
                key = txt_feats,
                value = txt_feats,
                key_padding_mask = padding_mask
            )
            
        fused_feats = self.ln(img_feats + attn_output)
        
        cls_feat = fused_feats[:, 0, :]
        cls_feat = self.dropout(cls_feat)
        
        logits_specific = self.classifier_specific(cls_feat)

        return logits_specific
    
if __name__ == "__main__":
    model = MultiModalNet_NoRegion()

    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_txt = ["Lung opacity found.", "No findings."]

    try:
        out_spec = model(dummy_img, dummy_txt)
        out_spec_vis = model(dummy_img, ["", ""])
        print(f"输入图片: {dummy_img.shape}")
        print(f"多模态输出结果: {out_spec.shape}")
        print(f"纯视觉输出结果: {out_spec_vis.shape}")
        if out_spec.shape == (2, 30):
            print("单任务模型测试通过！")
        else:
            print("输出形状不对")
            
    except Exception as e:
        print(f"运行报错: {e}")