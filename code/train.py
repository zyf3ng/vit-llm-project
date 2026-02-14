import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import Subset
import random

from dataset import ChestXrayDataset
from model import MultiModalNet

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 100
PATIENCE = 10
NUM_WORKERS = 4

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_CSV = os.path.join(CURRENT_DIR, '..', 'archive', 'indiana_reports.csv')
LABEL_CSV = os.path.join(CURRENT_DIR, '..', 'dataset_with_labels_2.csv')
IMG_DIR = os.path.join(CURRENT_DIR, '..', 'archive', 'images', 'images_normalized')
SAVE_DIR = os.path.join(CURRENT_DIR, '..', 'model_loss_21')

def plot_analysis(history, save_path):
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 辅助函数：给关键点打标签
    def annotate_points(ax, x, y, mode='max', color='red'):
        """
        mode: 'max' 找最大值标注 (适合 F1), 'min' 找最小值标注 (适合 Loss)
        """
        y = np.array(y)
        if mode == 'max':
            idx = np.argmax(y)
        else:
            idx = np.argmin(y)
            
        # 标注最佳值
        ax.annotate(f'{y[idx]:.4f}', (x[idx], y[idx]), 
                    xytext=(0, 10), textcoords='offset points', 
                    ha='center', fontsize=9, color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle="->", color=color))
        
        # 标注最后一个值 (如果它不是最佳值的话)
        if idx != len(x) - 1:
            last_x, last_y = x[-1], y[-1]
            ax.annotate(f'{last_y:.4f}', (last_x, last_y), 
                        xytext=(0, 10), textcoords='offset points', 
                        ha='center', fontsize=8, color='black')

    # --- 1. Total Loss ---
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', color='blue', alpha=0.6)
    ax.plot(epochs, history['val_loss'], label='Val Loss', color='red', linestyle='--')
    annotate_points(ax, epochs, history['val_loss'], mode='min') # 标出最低 Loss
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True)
    
    # --- 2. Total F1 ---
    ax = axes[0, 1]
    ax.plot(epochs, history['val_f1_total'], label='Val Total F1', color='red', marker='.')
    annotate_points(ax, epochs, history['val_f1_total'], mode='max') # 标出最高 F1
    ax.set_title('Total F1-Score')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    
    # --- 3. Spec Loss ---
    ax = axes[1, 0]
    ax.plot(epochs, history['train_spec'], label='Train Spec Loss', color='green', alpha=0.6)
    ax.plot(epochs, history['val_spec'], label='Val Spec Loss', color='orange', linestyle='--')
    annotate_points(ax, epochs, history['val_spec'], mode='min')
    ax.set_title('Specific Loss')
    ax.legend()
    ax.grid(True)
    
    # --- 4. Spec F1 ---
    ax = axes[1, 1]
    ax.plot(epochs, history['val_f1_spec'], label='Val Spec F1', color='orange', marker='.')
    annotate_points(ax, epochs, history['val_f1_spec'], mode='max')
    ax.set_title('Specific F1-Score')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    # --- 5. Reg Loss ---
    ax = axes[2, 0]
    ax.plot(epochs, history['train_reg'], label='Train Reg Loss', color='purple', alpha=0.6)
    ax.plot(epochs, history['val_reg'], label='Val Reg Loss', color='pink', linestyle='--')
    annotate_points(ax, epochs, history['val_reg'], mode='min')
    ax.set_title('Region Loss')
    ax.set_xlabel('Epochs')
    ax.legend()
    ax.grid(True)
    
    # --- 6. Reg F1 ---
    ax = axes[2, 1]
    ax.plot(epochs, history['val_f1_reg'], label='Val Reg F1', color='pink', marker='.')
    annotate_points(ax, epochs, history['val_f1_reg'], mode='max')
    ax.set_title('Region F1-Score')
    ax.set_xlabel('Epochs')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_epoch(model, loader, criterion, optimizer, device, dropout_prob):
    model.train()
    total_loss = 0
    total_spec = 0
    total_reg = 0
    
    loop = tqdm(loader, desc="Training", leave=False)
    
    for batch_imgs, batch_txts, batch_lbl_spec, batch_lbl_reg in loop:
        imgs = batch_imgs.to(device)
        lbl_spec = batch_lbl_spec.to(device)
        lbl_reg = batch_lbl_reg.to(device)
        
        final_txt_input = list(batch_txts) 
        
        #dropout_prob = 0.3
        
        for i in range(len(final_txt_input)):
            if random.random() < dropout_prob:
                final_txt_input[i] = "" 
        
        out_spec, out_reg = model(imgs, final_txt_input)
        
        loss_spec = criterion(out_spec, lbl_spec)
        loss_reg = criterion(out_reg, lbl_reg)
        loss = 2.0 * loss_spec + loss_reg
        
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
            loss = 2.0 * loss_spec + loss_reg
            
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
    
    model = MultiModalNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_val_spec_f1 = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(SAVE_DIR, "best_model.pth")
    plot_path = os.path.join(SAVE_DIR, "plot_analysis.png")

    history = {
        'train_loss': [], 'val_loss': [], 'val_f1_total': [],
        'train_spec': [], 'val_spec': [], 'val_f1_spec': [],
        'train_reg': [],  'val_reg': [],  'val_f1_reg': []
    }
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        if epoch < 20:
            current_dropout = 0.4 - (epoch / 20) * 0.3
        else:
            current_dropout = 0.1

        t_loss, t_spec, t_reg = train_epoch(model, train_loader, criterion, optimizer, DEVICE,current_dropout)
        
        v_loss, v_spec, v_reg, v_f1_t, v_f1_s, v_f1_r = eval_epoch(model, val_loader, criterion, DEVICE)    
        
        print(f"Total | Loss: {t_loss:.4f}(T) / {v_loss:.4f}(V) | F1: {v_f1_t:.4f}")
        print(f"Spec  | Loss: {t_spec:.4f}(T) / {v_spec:.4f}(V) | F1: {v_f1_s:.4f}")
        print(f"Reg   | Loss: {t_reg:.4f}(T) / {v_reg:.4f}(V) | F1: {v_f1_r:.4f}")
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_f1_total'].append(v_f1_t)
        
        history['train_spec'].append(t_spec)
        history['val_spec'].append(v_spec)
        history['val_f1_spec'].append(v_f1_s)
        
        history['train_reg'].append(t_reg)
        history['val_reg'].append(v_reg)
        history['val_f1_reg'].append(v_f1_r)
        
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

    print("\n===============test===============")
    model.load_state_dict(torch.load(best_model_path))
    t_loss, t_spec, t_reg, t_f1_t, t_f1_s, t_f1_r = eval_epoch(model, test_loader, criterion, DEVICE)
    print(f"Total | Loss: {t_loss:.4f} | F1: {t_f1_t:.4f}")
    print(f"Spec  | Loss: {t_spec:.4f} | F1: {t_f1_s:.4f}")
    print(f"Reg   | Loss: {t_reg:.4f} | F1: {t_f1_r:.4f}")

if __name__ == "__main__":
    main()