import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from dataset import ChestXrayDataset
from model import MultiModalNet

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 100
PATIENCE = 20
NUM_WORKERS = 4
alpha = 1.0
lam = 1.0

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_CSV = os.path.join(CURRENT_DIR, '..', 'archive', 'indiana_reports.csv')
LABEL_CSV = os.path.join(CURRENT_DIR, '..', 'dataset_with_labels_2.csv')
IMG_DIR = os.path.join(CURRENT_DIR, '..', 'archive', 'images', 'images_normalized')
SAVE_DIR = os.path.join(CURRENT_DIR, '..', 'model_loss_11')

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss_mm = 0
    total_spec_mm = 0
    total_reg_mm = 0
    total_loss_vis = 0
    
    # 预测结果
    preds_spec_mm, labels_spec = [], []
    preds_reg_mm, labels_reg = [], []
    
    preds_spec_vis = []
    preds_reg_vis = []

    with torch.no_grad():
        loop = tqdm(loader, desc="Validating", leave=False)
        for batch_imgs, batch_txts, batch_lbl_spec, batch_lbl_reg in loop:
            imgs = batch_imgs.to(device)
            lbl_spec = batch_lbl_spec.to(device)
            lbl_reg = batch_lbl_reg.to(device)
            
            # --- 1. 多模态 (MM) ---
            out_spec_mm, out_reg_mm = model(imgs, batch_txts)
            loss_spec_mm = criterion(out_spec_mm, lbl_spec)
            loss_reg_mm = criterion(out_reg_mm, lbl_reg)
            loss_mm = alpha * loss_spec_mm + loss_reg_mm
            
            total_loss_mm += loss_mm.item()
            total_spec_mm += loss_spec_mm.item()
            total_reg_mm += loss_reg_mm.item()

            preds_spec_mm.append((torch.sigmoid(out_spec_mm) > 0.5).float().cpu().numpy())
            preds_reg_mm.append((torch.sigmoid(out_reg_mm) > 0.5).float().cpu().numpy())
            
            labels_spec.append(lbl_spec.cpu().numpy())
            labels_reg.append(lbl_reg.cpu().numpy())
            
            # --- 2. 纯视觉 (Vis) ---
            empty_txts = [""] * len(batch_txts)
            out_spec_vis, out_reg_vis = model(imgs, empty_txts)
            
            loss_spec_vis = criterion(out_spec_vis, lbl_spec)
            loss_reg_vis = criterion(out_reg_vis, lbl_reg)
            loss_vis = alpha * loss_spec_vis + loss_reg_vis 
            
            total_loss_vis += loss_vis.item()

            preds_spec_vis.append((torch.sigmoid(out_spec_vis) > 0.5).float().cpu().numpy())
            preds_reg_vis.append((torch.sigmoid(out_reg_vis) > 0.5).float().cpu().numpy()) # 收集区域预测

    n = len(loader)
    
    avg_loss_mm = total_loss_mm / n
    avg_spec_mm = total_spec_mm / n
    avg_reg_mm = total_reg_mm / n
    
    L_spec = np.vstack(labels_spec)
    L_reg = np.vstack(labels_reg)
    
    # MM Metrics
    P_spec_mm = np.vstack(preds_spec_mm)
    P_reg_mm = np.vstack(preds_reg_mm)
    f1_s_mm = f1_score(L_spec, P_spec_mm, average='micro')
    f1_r_mm = f1_score(L_reg, P_reg_mm, average='micro')
    
    P_total_mm = np.hstack([P_spec_mm, P_reg_mm])
    L_total = np.hstack([L_spec, L_reg])
    f1_t_mm = f1_score(L_total, P_total_mm, average='micro')
    
    # Vis Metrics (新增 Reg)
    P_spec_vis = np.vstack(preds_spec_vis)
    P_reg_vis = np.vstack(preds_reg_vis) # 堆叠数组
    
    f1_s_vis = f1_score(L_spec, P_spec_vis, average='micro')
    f1_r_vis = f1_score(L_reg, P_reg_vis, average='micro') # 计算视觉对区域的 F1
    
    # 返回 8 个值
    return avg_loss_mm, avg_spec_mm, avg_reg_mm, f1_t_mm, f1_s_mm, f1_r_mm, f1_s_vis, f1_r_vis

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
        
    print(f"使用设备: {DEVICE}")
    
    train_ds_full = ChestXrayDataset(REPORT_CSV, LABEL_CSV, IMG_DIR, split='train')
    val_ds_full = ChestXrayDataset(REPORT_CSV, LABEL_CSV, IMG_DIR, split='val')

    total_len = len(train_ds_full)
    print(f"数据总数: {total_len}")
    
    train_size = int(0.7 * total_len)
    val_size = int(0.2 * total_len) 
    test_size = total_len - train_size - val_size 
    
    print(f"训练集: {train_size} 张")
    print(f"验证集: {val_size} 张")
    print(f"测试集: {test_size} 张")
    
    generator = torch.Generator().manual_seed(37)
    
    train_sub, val_sub, test_sub = random_split(
        train_ds_full, 
        [train_size, val_size, test_size],
        generator=generator
    )

    train_dataset = train_sub 
    val_dataset = Subset(val_ds_full, val_sub.indices)
    test_dataset = Subset(val_ds_full, test_sub.indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    model = MultiModalNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    
    best_model_path = os.path.join(SAVE_DIR, "best_model.pth")

    print("\n===============Test===============")
    model.load_state_dict(torch.load(best_model_path))
    t_loss, t_spec, t_reg, t_f1_t, t_f1_s, t_f1_r, t_f1_s_vis, t_f1_r_vis = eval_epoch(model, test_loader, criterion, DEVICE)    
        
    print(f"Total(MM) | Loss: {t_loss:.4f} | F1: {t_f1_t:.4f}")
    print(f"Spec (MM) | Loss: {t_spec:.4f} | F1: {t_f1_s:.4f}")
    print(f"Reg  (MM) | Loss: {t_reg:.4f}  | F1: {t_f1_r:.4f}")
    print(f"Spec (Vis)| F1: {t_f1_s_vis:.4f}")
    print(f"Reg  (Vis)| F1: {t_f1_r_vis:.4f}")

if __name__ == "__main__":
    main()