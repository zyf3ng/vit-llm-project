import torch
import torch.nn as nn
from image_encoder import ImageEncoder
from text_encoder import TextEncoder

class MultiModalNet(nn.Module):
    def __init__(self, num_specific=30, num_region=5, d_model=768, num_heads=8, dropout=0.1, num_experts=3):
        super().__init__()

        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        
        # 保留极其强大的交叉注意力提取特征
        self.cross_attention = nn.MultiheadAttention(
            embed_dim = d_model, 
            num_heads = num_heads, 
            dropout = dropout, 
            batch_first = True
        )
        
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # ========= MMOE 核心组件开始 =========
        self.num_experts = num_experts
        # 定义多个专家 (Experts)
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        ) for _ in range(self.num_experts)])
        
        # 定义两个任务各自的门控网络 (Gates)
        self.gate_specific = nn.Linear(d_model, self.num_experts)
        self.gate_region = nn.Linear(d_model, self.num_experts)
        # ========= MMOE 核心组件结束 =========

        self.classifier_specific = nn.Linear(d_model, num_specific)
        self.classifier_region = nn.Linear(d_model, num_region)
        
    def forward(self, images, text_list=None):
        # 1. 提取视觉特征
        img_feats = self.image_encoder(images)
        
        # 2. 视觉单模态约束的判断逻辑
        is_pure_vision = False
        if text_list is None:
            is_pure_vision = True
        elif isinstance(text_list, list) and all(t == "" for t in text_list):
            is_pure_vision = True
            
        # 3. 跨模态特征融合
        if is_pure_vision:
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
        
        # ========= MMOE 动态路由逻辑 =========
        # 1. 让所有专家（Expert）针对特征给出意见 -> shape: (B, num_experts, d_model)
        expert_outputs = torch.stack([expert(cls_feat) for expert in self.experts], dim=1)
        
        # 2. 具体疾病分类任务 (Specific) 的门控路由
        # 计算特定任务对每个专家的注意力权重
        gate_spec_weights = torch.softmax(self.gate_specific(cls_feat), dim=-1).unsqueeze(-1) # (B, num_experts, 1)
        feat_specific = torch.sum(expert_outputs * gate_spec_weights, dim=1) # 加权求和得到专属特征
        
        # 3. 区域定位任务 (Region) 的门控路由
        gate_reg_weights = torch.softmax(self.gate_region(cls_feat), dim=-1).unsqueeze(-1) # (B, num_experts, 1)
        feat_region = torch.sum(expert_outputs * gate_reg_weights, dim=1)
        
        # ========= 最终分类 =========
        logits_specific = self.classifier_specific(feat_specific)
        logits_region = self.classifier_region(feat_region)
        
        return logits_specific, logits_region