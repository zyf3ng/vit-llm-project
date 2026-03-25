import torch
import torch.nn as nn
from image_encoder import ImageEncoder
from text_encoder import TextEncoder

class MultiModalNet(nn.Module):
    def __init__(self, num_specific=30, num_region=5, d_model=768, dropout=0.1):
        super().__init__()

        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        
        # ========= 极其关键的修改：投影对齐层 =========
        # 将视觉和文本分别映射到共享的 768 维多模态空间
        self.img_proj = nn.Linear(d_model, d_model)
        self.txt_proj = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        # ===============================================

        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.classifier_specific = nn.Linear(d_model, num_specific)
        self.classifier_region = nn.Linear(d_model, num_region)
        
    def forward(self, images, text_list=None):
        # 1. 提取视觉特征
        img_feats = self.image_encoder(images)
        img_cls = img_feats[:, 0, :] 
        
        is_pure_vision = False
        if text_list is None:
            is_pure_vision = True
        elif isinstance(text_list, list) and all(t == "" for t in text_list):
            is_pure_vision = True
            
        if is_pure_vision:
            # 纯视觉时，直接过投影层
            fused_cls = self.activation(self.img_proj(img_cls))
        else:
            # 2. 提取文本特征
            txt_feats, txt_mask = self.text_encoder(text_list)
            txt_cls = txt_feats[:, 0, :] 
            
            # 3. 极其严谨的多模态对齐与融合
            # 先各自投影到共享空间
            aligned_img = self.activation(self.img_proj(img_cls))
            aligned_txt = self.activation(self.txt_proj(txt_cls))
            
            # 此时再做元素级相乘，物理意义就极其完美了！
            fused_cls = aligned_img * aligned_txt 

        fused_cls = self.ln(fused_cls)
        cls_feat = self.dropout(fused_cls)
        
        logits_specific = self.classifier_specific(cls_feat)
        logits_region = self.classifier_region(cls_feat)
        
        return logits_specific, logits_region