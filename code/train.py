import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import random

from dataset import ChestXrayDataset
from model import MultiModalNet

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def seed_everything(seed=37):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def plot_analysis(history, save_path):
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    
    def annotate_points(ax, x, y, mode='max', color='red'):
        if len(y) == 0: return # 防止空数据报错
        y = np.array(y)
        if mode == 'max':
            idx = np.argmax(y)
        else:
            idx = np.argmin(y)
        ax.annotate(f'{y[idx]:.4f}', (x[idx], y[idx]), 
                    xytext=(0, 10), textcoords='offset points', 
                    ha='center', fontsize=9, color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle="->", color=color))
        if idx != len(x) - 1:
            last_x, last_y = x[-1], y[-1]
            ax.annotate(f'{last_y:.4f}', (last_x, last_y), 
                        xytext=(0, 10), textcoords='offset points', 
                        ha='center', fontsize=8, color='black')

    # --- 1. Total (MM) ---
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', color='blue', alpha=0.6)
    ax.plot(epochs, history['val_loss'], label='Val Loss', color='red', linestyle='--')
    annotate_points(ax, epochs, history['val_loss'], mode='min')
    ax.set_title('Total Loss (Multimodal)')
    ax.legend()
    ax.grid(True)
    
    ax = axes[0, 1]
    ax.plot(epochs, history['val_f1_total'], label='Val Total F1', color='red', marker='.')
    annotate_points(ax, epochs, history['val_f1_total'], mode='max')
    ax.set_title('Total F1 (Multimodal)')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    
    # --- 2. Spec (MM) ---
    ax = axes[1, 0]
    ax.plot(epochs, history['train_spec'], label='Train Spec', color='green', alpha=0.6)
    ax.plot(epochs, history['val_spec'], label='Val Spec', color='orange', linestyle='--')
    annotate_points(ax, epochs, history['val_spec'], mode='min')
    ax.set_title('Specific Loss (Multimodal)')
    ax.legend()
    ax.grid(True)
    
    ax = axes[1, 1]
    ax.plot(epochs, history['val_f1_spec'], label='Val Spec F1', color='orange', marker='.')
    annotate_points(ax, epochs, history['val_f1_spec'], mode='max')
    ax.set_title('Specific F1 (Multimodal)')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    # --- 3. Reg (MM) ---
    ax = axes[2, 0]
    ax.plot(epochs, history['train_reg'], label='Train Reg', color='purple', alpha=0.6)
    ax.plot(epochs, history['val_reg'], label='Val Reg', color='pink', linestyle='--')
    annotate_points(ax, epochs, history['val_reg'], mode='min')
    ax.set_title('Region Loss (Multimodal)')
    ax.legend()
    ax.grid(True)
    
    ax = axes[2, 1]
    ax.plot(epochs, history['val_f1_reg'], label='Val Reg F1', color='pink', marker='.')
    annotate_points(ax, epochs, history['val_f1_reg'], mode='max')
    ax.set_title('Region F1 (Multimodal)')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    
    # --- 4. Vision Only ---
    # 左边画 Spec (具体疾病) 的纯视觉 F1
    ax = axes[3, 0]
    ax.plot(epochs, history['val_f1_vis_spec'], label='Vis Only Spec F1', color='brown', marker='o')
    annotate_points(ax, epochs, history['val_f1_vis_spec'], mode='max')
    ax.set_title('Vision Only - Specific F1')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    
    # 右边画 Reg (区域) 的纯视觉 F1
    ax = axes[3, 1]
    ax.plot(epochs, history['val_f1_vis_reg'], label='Vis Only Reg F1', color='brown', marker='o')
    annotate_points(ax, epochs, history['val_f1_vis_reg'], mode='max')
    ax.set_title('Vision Only - Region F1')
    ax.set_xlabel('Epochs')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_spec = 0
    total_reg = 0
    
    loop = tqdm(loader, desc="Training", leave=False)
    
    for batch_imgs, batch_txts, batch_lbl_spec, batch_lbl_reg in loop:
        imgs = batch_imgs.to(device)
        lbl_spec = batch_lbl_spec.to(device)
        lbl_reg = batch_lbl_reg.to(device)
        
        out_spec_mm, out_reg_mm = model(imgs, batch_txts)
        
        loss_spec_mm = criterion(out_spec_mm, lbl_spec)
        loss_reg_mm = criterion(out_reg_mm, lbl_reg)
        loss_mm = alpha * loss_spec_mm + loss_reg_mm
        
        empty_txts = [""] * len(batch_txts) 
        out_spec_vis, out_reg_vis = model(imgs, empty_txts)
        
        loss_spec_vis = criterion(out_spec_vis, lbl_spec)
        loss_reg_vis = criterion(out_reg_vis, lbl_reg)
        loss_vis = alpha * loss_spec_vis + loss_reg_vis
        
        loss = loss_mm + lam * loss_vis
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_spec += loss_spec_mm.item()
        total_reg += loss_reg_mm.item()
        
        loop.set_postfix(loss=loss.item())
    
    n = len(loader)   
    return total_loss/n, total_spec/n, total_reg/n

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss_mm = 0
    total_spec_mm = 0
    total_reg_mm = 0
    total_loss_vis = 0
    
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
            preds_reg_vis.append((torch.sigmoid(out_reg_vis) > 0.5).float().cpu().numpy())

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
    
    # Vis Metrics
    P_spec_vis = np.vstack(preds_spec_vis)
    P_reg_vis = np.vstack(preds_reg_vis)
    
    f1_s_vis = f1_score(L_spec, P_spec_vis, average='micro')
    f1_r_vis = f1_score(L_reg, P_reg_vis, average='micro')
    
    # 返回 8 个值
    return avg_loss_mm, avg_spec_mm, avg_reg_mm, f1_t_mm, f1_s_mm, f1_r_mm, f1_s_vis, f1_r_vis

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    seed_everything(37)

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

    frozen_layers = 0
    trainable_layers = 0
    
    for name, param in model.image_encoder.named_parameters():
        param.requires_grad = False
    
        if "layer.9" in name or "layer.10" in name or "layer.11" in name or "layernorm" in name or "pooler" in name:
            param.requires_grad = True
            trainable_layers += 1
        else:
            frozen_layers += 1

    #optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    best_val_spec_f1 = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(SAVE_DIR, "best_model.pth")
    plot_path = os.path.join(SAVE_DIR, "plot_analysis.png")

    history = {
        'train_loss': [], 'val_loss': [], 'val_f1_total': [],
        'train_spec': [], 'val_spec': [], 'val_f1_spec': [],
        'train_reg': [],  'val_reg': [],  'val_f1_reg': [],
        'val_f1_vis_spec': [], 'val_f1_vis_reg': []
    }
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

        t_loss, t_spec, t_reg = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        v_loss, v_spec, v_reg, v_f1_t, v_f1_s, v_f1_r, v_f1_s_vis, v_f1_r_vis = eval_epoch(model, val_loader, criterion, DEVICE)    
        
        print(f"Total(MM) | Loss: {t_loss:.4f}(T) / {v_loss:.4f}(V) | F1: {v_f1_t:.4f}")
        print(f"Spec (MM) | Loss: {t_spec:.4f}(T) / {v_spec:.4f}(V) | F1: {v_f1_s:.4f}")
        print(f"Reg  (MM) | Loss: {t_reg:.4f}(T) / {v_reg:.4f}(V) | F1: {v_f1_r:.4f}")
        print(f"Spec (Vis)| F1: {v_f1_s_vis:.4f}")
        print(f"Reg  (Vis)| F1: {v_f1_r_vis:.4f}")
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_f1_total'].append(v_f1_t)
        
        history['train_spec'].append(t_spec)
        history['val_spec'].append(v_spec)
        history['val_f1_spec'].append(v_f1_s)
        
        history['train_reg'].append(t_reg)
        history['val_reg'].append(v_reg)
        history['val_f1_reg'].append(v_f1_r)

        history['val_f1_vis_spec'].append(v_f1_s_vis)
        history['val_f1_vis_reg'].append(v_f1_r_vis)
        
        plot_analysis(history, plot_path)

        if v_f1_s > best_val_spec_f1:
            best_val_spec_f1 = v_f1_s
            torch.save(model.state_dict(), best_model_path)
            print(f"验证集表现提升(MM)，已保存最佳模型!")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"loss未下降 ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\n触发早停，训练结束。")
                break

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