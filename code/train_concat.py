import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

from dataset import ChestXrayDataset
from model_concat import MultiModalNet_Concat

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LEARNING_RATE = 5e-4 
NUM_EPOCHS = 100 
PATIENCE = 10
NUM_WORKERS = 4

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_CSV = os.path.join(CURRENT_DIR, '..', 'archive', 'indiana_reports.csv')
LABEL_CSV = os.path.join(CURRENT_DIR, '..', 'dataset_with_labels_2.csv')
IMG_DIR = os.path.join(CURRENT_DIR, '..', 'archive', 'images', 'images_normalized')
SAVE_DIR = os.path.join(CURRENT_DIR, '..', 'model_concat')

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
        loss = 4.0 * loss_spec + loss_reg
        
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
            loss = 4.0 * loss_spec + loss_reg
            
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
    
    full_dataset = ChestXrayDataset(REPORT_CSV, LABEL_CSV, IMG_DIR)
    total_len = len(full_dataset)
    print(f"数据总数: {total_len}")
    
    train_size = int(0.7 * total_len)
    val_size = int(0.2 * total_len) 
    test_size = total_len - train_size - val_size 
    
    print(f"训练集: {train_size} 张")
    print(f"验证集: {val_size} 张")
    print(f"测试集: {test_size} 张")
    
    generator = torch.Generator().manual_seed(37)
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    model = MultiModalNet_Concat().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_val_spec_f1 = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(SAVE_DIR, "best_model.pth")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        t_loss, t_spec, t_reg = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

        v_loss, v_spec, v_reg, v_f1_t,v_f1_s, v_f1_r = eval_epoch(model, val_loader, criterion, DEVICE)    
        
        print(f"Total | Loss: {t_loss:.4f}(T) / {v_loss:.4f}(V) | F1: {v_f1_t:.4f}")
        print(f"Spec  | Loss: {t_spec:.4f}(T) / {v_spec:.4f}(V) | F1: {v_f1_s:.4f}")
        print(f"Reg   | Loss: {t_reg:.4f}(T) / {v_reg:.4f}(V) | F1: {v_f1_r:.4f}")
        
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