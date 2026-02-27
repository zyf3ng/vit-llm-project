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
from model_no_region import MultiModalNet_NoRegion

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 100
PATIENCE = 20
NUM_WORKERS = 4

# 完全对齐！主线任务乘2，纯视觉单模态惩罚1.5
alpha = 2.0
lam = 1.5

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_CSV = os.path.join(CURRENT_DIR, '..', 'archive', 'indiana_reports.csv')
LABEL_CSV = os.path.join(CURRENT_DIR, '..', 'dataset_with_labels_2.csv')
IMG_DIR = os.path.join(CURRENT_DIR, '..', 'archive', 'images', 'images_normalized')
SAVE_DIR = os.path.join(CURRENT_DIR, '..', 'model_no_region')

def plot_analysis(history, save_path):
    epochs = list(range(1, len(history['train_loss']) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    def annotate_points(ax, x, y, mode='max', color='red'):
        if len(y) == 0: return 
        y = np.array(y)
        idx = np.argmax(y) if mode == 'max' else np.argmin(y)
        ax.annotate(f'{y[idx]:.4f}', (x[idx], y[idx]), 
                    xytext=(0, 10), textcoords='offset points', 
                    ha='center', fontsize=9, color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle="->", color=color))
        if idx != len(x) - 1:
            ax.annotate(f'{y[-1]:.4f}', (x[-1], y[-1]), 
                        xytext=(0, 10), textcoords='offset points', 
                        ha='center', fontsize=8, color='black')

    # Loss 
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    ax.plot(epochs, history['val_loss'], label='Val Loss', color='red', linestyle='--')
    annotate_points(ax, epochs, history['val_loss'], mode='min')
    ax.set_title('Loss (No Region)')
    ax.legend(); ax.grid(True)
    
    # MM F1
    ax = axes[0, 1]
    ax.plot(epochs, history['val_f1_spec_mm'], label='Val Spec F1 (MM)', color='orange', marker='.')
    annotate_points(ax, epochs, history['val_f1_spec_mm'], mode='max')
    ax.set_title('Multimodal F1')
    ax.set_ylim(0, 1)
    ax.legend(); ax.grid(True)

    # Vis F1
    ax = axes[1, 0]
    ax.plot(epochs, history['val_f1_spec_vis'], label='Val Spec F1 (Vis)', color='brown', marker='o')
    annotate_points(ax, epochs, history['val_f1_spec_vis'], mode='max')
    ax.set_title('Vision Only F1')
    ax.set_ylim(0, 1)
    ax.legend(); ax.grid(True)
    
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc="Training", leave=False)
    
    for batch_imgs, batch_txts, batch_lbl_spec, _ in loop:
        imgs = batch_imgs.to(device)
        lbl_spec = batch_lbl_spec.to(device)
        
        # 多模态主线
        out_spec_mm = model(imgs, batch_txts)
        loss_mm = alpha * criterion(out_spec_mm, lbl_spec)
        
        # 纯视觉惩罚
        empty_txts = [""] * len(batch_txts)
        out_spec_vis = model(imgs, empty_txts)
        loss_vis = alpha * criterion(out_spec_vis, lbl_spec)
        
        loss = loss_mm + lam * loss_vis
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    n = len(loader)   
    return total_loss/n

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds_spec_mm, preds_spec_vis, labels_spec = [], [], []

    with torch.no_grad():
        loop = tqdm(loader, desc="Validating", leave=False)
        for batch_imgs, batch_txts, batch_lbl_spec, _ in loop:
            imgs = batch_imgs.to(device)
            lbl_spec = batch_lbl_spec.to(device)
            
            # 多模态
            out_spec_mm = model(imgs, batch_txts)
            loss_mm = alpha * criterion(out_spec_mm, lbl_spec)
            
            # 纯视觉
            empty_txts = [""] * len(batch_txts)
            out_spec_vis = model(imgs, empty_txts)
            loss_vis = alpha * criterion(out_spec_vis, lbl_spec)
            
            total_loss += (loss_mm + lam * loss_vis).item()

            preds_spec_mm.append((torch.sigmoid(out_spec_mm) > 0.5).float().cpu().numpy())
            preds_spec_vis.append((torch.sigmoid(out_spec_vis) > 0.5).float().cpu().numpy())
            labels_spec.append(lbl_spec.cpu().numpy())

    n = len(loader)
    L_spec = np.vstack(labels_spec)
    
    f1_s_mm = f1_score(L_spec, np.vstack(preds_spec_mm), average='micro')
    f1_s_vis = f1_score(L_spec, np.vstack(preds_spec_vis), average='micro')
    
    return total_loss/n, f1_s_mm, f1_s_vis

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"使用设备: {DEVICE}")
    
    train_ds_full = ChestXrayDataset(REPORT_CSV, LABEL_CSV, IMG_DIR, split='train')
    val_ds_full = ChestXrayDataset(REPORT_CSV, LABEL_CSV, IMG_DIR, split='val')
    total_len = len(train_ds_full)
    
    train_size = int(0.7 * total_len)
    val_size = int(0.2 * total_len) 
    test_size = total_len - train_size - val_size 
    
    generator = torch.Generator().manual_seed(37) # 锁死 37！
    train_sub, val_sub, test_sub = random_split(train_ds_full, [train_size, val_size, test_size], generator=generator)

    train_dataset = train_sub 
    val_dataset = Subset(val_ds_full, val_sub.indices)
    test_dataset = Subset(val_ds_full, test_sub.indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    model = MultiModalNet_NoRegion().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    for name, param in model.image_encoder.named_parameters():
        param.requires_grad = False
        if "layer.9" in name or "layer.10" in name or "layer.11" in name or "layernorm" in name or "pooler" in name:
            param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    best_combined_score = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(SAVE_DIR, "best_model.pth")
    plot_path = os.path.join(SAVE_DIR, "plot_analysis_noregion.png")
    
    history = {'train_loss': [], 'val_loss': [], 'val_f1_spec_mm': [], 'val_f1_spec_vis': []}
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        t_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        v_loss, v_f1_mm, v_f1_vis = eval_epoch(model, val_loader, criterion, DEVICE)    
        
        print(f"Loss | Train: {t_loss:.4f} | Val: {v_loss:.4f}")
        print(f"Spec | MM F1: {v_f1_mm:.4f} | Vis F1: {v_f1_vis:.4f}")
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_f1_spec_mm'].append(v_f1_mm)
        history['val_f1_spec_vis'].append(v_f1_vis)
        
        plot_analysis(history, plot_path)
        
        combined_score = 0.6 * v_f1_mm + 0.4 * v_f1_vis
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            torch.save(model.state_dict(), best_model_path)
            print(f"验证集表现提升，综合得分 {combined_score:.4f}，已保存最佳模型!")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Loss 未下降 ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\n触发早停，训练结束。")
                break

    print("\n===============Test===============")
    model.load_state_dict(torch.load(best_model_path))
    t_loss, t_f1_mm, t_f1_vis = eval_epoch(model, test_loader, criterion, DEVICE)
    print(f"Test Spec (MM)  | F1: {t_f1_mm:.4f}")
    print(f"Test Spec (Vis) | F1: {t_f1_vis:.4f}")

if __name__ == "__main__":
    main()