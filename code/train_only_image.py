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
from model_only_image import MultiModalNet_OnlyImage 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 100
PATIENCE = 20
NUM_WORKERS = 4
alpha = 1.0 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_CSV = os.path.join(CURRENT_DIR, '..', 'archive', 'indiana_reports.csv')
LABEL_CSV = os.path.join(CURRENT_DIR, '..', 'dataset_with_labels_2.csv')
IMG_DIR = os.path.join(CURRENT_DIR, '..', 'archive', 'images', 'images_normalized')
SAVE_DIR = os.path.join(CURRENT_DIR, '..', 'model_only_image')

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

    # 1. Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', color='blue', alpha=0.6)
    ax.plot(epochs, history['val_loss'], label='Val Loss', color='red', linestyle='--')
    annotate_points(ax, epochs, history['val_loss'], mode='min')
    ax.set_title(f'Baseline Loss (alpha={alpha})')
    ax.legend()
    ax.grid(True)
    
    # 2. Spec F1
    ax = axes[0, 1]
    ax.plot(epochs, history['val_f1_spec'], label='Val Spec F1', color='orange', marker='.')
    annotate_points(ax, epochs, history['val_f1_spec'], mode='max')
    ax.set_title('Baseline Spec F1')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    
    # 3. Reg F1
    ax = axes[1, 0]
    ax.plot(epochs, history['val_f1_reg'], label='Val Reg F1', color='pink', marker='.')
    annotate_points(ax, epochs, history['val_f1_reg'], mode='max')
    ax.set_title('Baseline Reg F1')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    # 4. Total F1
    ax = axes[1, 1]
    ax.plot(epochs, history['val_f1_total'], label='Val Total F1', color='brown', marker='.')
    annotate_points(ax, epochs, history['val_f1_total'], mode='max')
    ax.set_title('Baseline Total F1')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# === Train Epoch ===
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
        
        out_spec, out_reg = model(imgs, batch_txts)
        
        loss_spec = criterion(out_spec, lbl_spec)
        loss_reg = criterion(out_reg, lbl_reg)  
        loss = alpha * loss_spec + loss_reg
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_spec += loss_spec.item()
        total_reg += loss_reg.item()
        
        loop.set_postfix(loss=loss.item())
    
    n = len(loader)   
    return total_loss/n, total_spec/n, total_reg/n

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_spec = 0
    total_reg = 0
    
    preds_spec, labels_spec = [], []
    preds_reg, labels_reg = [], []

    with torch.no_grad():
        loop = tqdm(loader, desc="Validating", leave=False)
        for batch_imgs, batch_txts, batch_lbl_spec, batch_lbl_reg in loop:
            imgs = batch_imgs.to(device)
            lbl_spec = batch_lbl_spec.to(device)
            lbl_reg = batch_lbl_reg.to(device)
            
            out_spec, out_reg = model(imgs, batch_txts)
            
            loss_spec = criterion(out_spec, lbl_spec)
            loss_reg = criterion(out_reg, lbl_reg)
            loss = alpha * loss_spec + loss_reg
            
            total_loss += loss.item()
            total_spec += loss_spec.item()
            total_reg += loss_reg.item()

            preds_spec.append((torch.sigmoid(out_spec) > 0.5).float().cpu().numpy())
            preds_reg.append((torch.sigmoid(out_reg) > 0.5).float().cpu().numpy())
            
            labels_spec.append(lbl_spec.cpu().numpy())
            labels_reg.append(lbl_reg.cpu().numpy())

    n = len(loader)
    
    avg_loss = total_loss / n
    avg_spec = total_spec / n
    avg_reg = total_reg / n
    
    L_spec = np.vstack(labels_spec)
    L_reg = np.vstack(labels_reg)
    P_spec = np.vstack(preds_spec)
    P_reg = np.vstack(preds_reg)
    
    f1_s = f1_score(L_spec, P_spec, average='micro')
    f1_r = f1_score(L_reg, P_reg, average='micro')
    
    P_total = np.hstack([P_spec, P_reg])
    L_total = np.hstack([L_spec, L_reg])
    f1_t = f1_score(L_total, P_total, average='micro')
    
    return avg_loss, avg_spec, avg_reg, f1_t, f1_s, f1_r

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
    
    model = MultiModalNet_OnlyImage().to(DEVICE)
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

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    best_val_spec_f1 = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(SAVE_DIR, "best_model.pth")
    plot_path = os.path.join(SAVE_DIR, "plot_analysis.png")

    history = {
        'train_loss': [], 'val_loss': [], 
        'val_f1_spec': [], 'val_f1_reg': [], 'val_f1_total': []
    }
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

        t_loss, t_spec, t_reg = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        v_loss, v_spec, v_reg, v_f1_t, v_f1_s, v_f1_r = eval_epoch(model, val_loader, criterion, DEVICE)    
        
        print(f"Total | Loss: {t_loss:.4f}(T) / {v_loss:.4f}(V) | F1: {v_f1_t:.4f}")
        print(f"Spec  | Loss: {t_spec:.4f}(T) / {v_spec:.4f}(V) | F1: {v_f1_s:.4f}")
        print(f"Reg   | Loss: {t_reg:.4f}(T) / {v_reg:.4f}(V) | F1: {v_f1_r:.4f}")
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_f1_spec'].append(v_f1_s)
        history['val_f1_reg'].append(v_f1_r)
        history['val_f1_total'].append(v_f1_t)
        
        plot_analysis(history, plot_path)

        if v_f1_s > best_val_spec_f1:
            best_val_spec_f1 = v_f1_s
            torch.save(model.state_dict(), best_model_path)
            print(f"验证集表现提升，已保存最佳模型!")

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
    v_loss, v_spec, v_reg, v_f1_t, v_f1_s, v_f1_r = eval_epoch(model, test_loader, criterion, DEVICE)    
        
    print(f"Total | Loss: {t_loss:.4f} | F1: {v_f1_t:.4f}")
    print(f"Spec  | Loss: {t_spec:.4f} | F1: {v_f1_s:.4f}")
    print(f"Reg   | Loss: {t_reg:.4f} | F1: {v_f1_r:.4f}")

if __name__ == "__main__":
    main()