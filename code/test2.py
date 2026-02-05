import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import ChestXrayDataset
from model import MultiModalNet 

# ================= 配置区域 =================
# 显卡设置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 4

# 路径 (根据你的实际路径修改)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_CSV = os.path.join(CURRENT_DIR, '..', 'archive', 'indiana_reports.csv')
LABEL_CSV = os.path.join(CURRENT_DIR, '..', 'dataset_with_labels_2.csv')
IMG_DIR = os.path.join(CURRENT_DIR, '..', 'archive', 'images', 'images_normalized')

# 🔥 模型权重路径 (改成你那个 5e-5 或 5e-4 的 best_model.pth)
BEST_MODEL_PATH = os.path.join(CURRENT_DIR, '..', 'model_loss_41_2', 'best_model.pth')
# ===========================================

# 1. 这里是你刚才要求的 eval_image_shuffle 函数
def eval_image_shuffle(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_spec = 0
    total_reg = 0
    
    preds_spec, labels_spec = [], []
    preds_reg, labels_reg = [], []

    
    with torch.no_grad():
        # 这里的 desc 改一下，方便你看进度
        loop = tqdm(loader, desc="Testing", leave=True)
        for batch_imgs, batch_txts, batch_lbl_spec, batch_lbl_reg in loop:
            imgs = batch_imgs.to(device)
            lbl_spec = batch_lbl_spec.to(device)
            lbl_reg = batch_lbl_reg.to(device)
            
            # 🔥 核心修改：图片错位 (Batch 内向后移一位)
            shuffled_imgs = torch.roll(imgs, shifts=1, dims=0)
            
            # 喂给模型：错图 + 对文
            out_spec, out_reg = model(shuffled_imgs, batch_txts)
            
            # 计算 Loss
            loss_spec = criterion(out_spec, lbl_spec)
            loss_reg = criterion(out_reg, lbl_reg)
            loss = 4.0 * loss_spec + loss_reg  # 保持和你训练时一样的权重
            
            total_loss += loss.item()
            total_spec += loss_spec.item()
            total_reg += loss_reg.item()

            preds_spec.append((torch.sigmoid(out_spec) > 0.5).float().cpu().numpy())
            labels_spec.append(lbl_spec.cpu().numpy())
            
            preds_reg.append((torch.sigmoid(out_reg) > 0.5).float().cpu().numpy())
            labels_reg.append(lbl_reg.cpu().numpy())

    n = len(loader)
    avg_loss = total_loss / n
    avg_spec = total_spec / n
    avg_reg = total_reg / n
    
    P_spec = np.vstack(preds_spec)
    L_spec = np.vstack(labels_spec)
    P_reg = np.vstack(preds_reg)
    L_reg = np.vstack(labels_reg)
    
    # 计算 F1
    f1_s = f1_score(L_spec, P_spec, average='micro')
    f1_r = f1_score(L_reg, P_reg, average='micro')
    
    P_total = np.hstack([P_spec, P_reg])
    L_total = np.hstack([L_spec, L_reg])
    f1_t = f1_score(L_total, P_total, average='micro')
    
    return avg_loss, avg_spec, avg_reg, f1_t, f1_s, f1_r

# 2. Main 函数：负责加载数据和模型，并【调用】上面的函数
def main():

    full_dataset = ChestXrayDataset(REPORT_CSV, LABEL_CSV, IMG_DIR)
    total_len = len(full_dataset)
    train_size = int(0.7 * total_len)
    val_size = int(0.2 * total_len)
    test_size = total_len - train_size - val_size
    

    _, _, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(37)
    )

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    

    model = MultiModalNet(num_specific=30, num_region=5).to(DEVICE)
    
    model.load_state_dict(torch.load(BEST_MODEL_PATH))



    criterion = nn.BCEWithLogitsLoss()


    val_loss, val_spec, val_reg, f1_t, f1_s, f1_r = eval_image_shuffle(model, test_loader, criterion, DEVICE)
    
    # --- E. 打印最终结果 ---
    print()
    print(f"Total F1  : {f1_t:.4f}")
    print(f"Spec F1   : {f1_s:.4f}")
    print(f"Reg F1    : {f1_r:.4f}")
    
    print("="*50)

if __name__ == "__main__":
    main()