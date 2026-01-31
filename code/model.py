import torch
import torch.nn as nn
from image_encoder import ImageEncoder
from text_encoder import TextEncoder

class MultiModalNet(nn.Module):
    def __init__(self, num_specific=30, num_region=5, d_model=768, num_heads=8, dropout=0.1):
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
        
        self.classifier_region = nn.Linear(d_model, num_region)
        
    def forward(self, images, text_list):
        img_feats = self.image_encoder(images)
        
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
        
        logits_region = self.classifier_region(cls_feat)
        
        return logits_specific, logits_region
    
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

    def forward(self, images, text_list):
        img_feats = self.image_encoder(images)
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
    # 准备假数据 (Batch Size = 2)
    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_txt = ["Lung opacity found.", "No findings."]
    
    print("="*40)
    print("🧪 测试 1: 原完整模型 (MultiModalNet)")
    print("="*40)
    try:
        model = MultiModalNet()
        out_spec, out_reg = model(dummy_img, dummy_txt)
        
        print(f"输入图片: {dummy_img.shape}")
        print(f"Specific Output: {out_spec.shape}")
        print(f"Region Output:   {out_reg.shape}")
        
        if out_spec.shape == (2, 30) and out_reg.shape == (2, 5):
            print("✅ 原模型测试通过！输出两个结果。")
        else:
            print("❌ 原模型输出形状不对！")
            
    except Exception as e:
        print(f"❌ 原模型运行报错: {e}")

    print("\n" + "="*40)
    print("🧪 测试 2: 消融模型 (MultiModalNet_NoRegion)")
    print("="*40)
    try:
        # 实例化消融模型
        model_ablation = MultiModalNet_NoRegion()
        
        # ⚠️ 注意：这里只能接收一个返回值
        output = model_ablation(dummy_img, dummy_txt)
        
        print(f"输入图片: {dummy_img.shape}")
        print(f"输出结果: {output.shape} (应该只有 Specific)")
        
        # 验证逻辑：必须是 (2, 30) 且不能是 tuple
        if isinstance(output, tuple):
            print("❌ 错误：模型依然返回了 Tuple，说明没改干净！")
        elif output.shape == (2, 30):
            print("✅ 消融模型测试通过！只输出了 Specific 分类结果。")
        else:
            print(f"❌ 形状错误：期望 (2, 30)，实际 {output.shape}")

    except Exception as e:
        print(f"❌ 消融模型运行报错: {e}")