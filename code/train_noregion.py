import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

from dataset import ChestXrayDataset
from model_noregion import MultiModalNet_NoRegion

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
SAVE_DIR = os.path.join(CURRENT_DIR, '..', 'model_no_region')

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    loop = tqdm(loader, desc="Training", leave=False)
    
    for batch_imgs, batch_txts, batch_lbl_spec, _ in loop:
        imgs = batch_imgs.to(device)
        lbl_spec = batch_lbl_spec.to(device)
        
        out_spec = model(imgs, batch_txts)
        
        loss = criterion(out_spec, lbl_spec)
        
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
    
    preds_spec, labels_spec = [], []

    with torch.no_grad():
        loop = tqdm(loader, desc="Validating", leave=False)
        for batch_imgs, batch_txts, batch_lbl_spec, _ in loop:
            imgs = batch_imgs.to(device)
            lbl_spec = batch_lbl_spec.to(device)
            
            out_spec = model(imgs, batch_txts)
            
            loss = criterion(out_spec, lbl_spec)
            total_loss += loss.item()

            preds_spec.append((torch.sigmoid(out_spec) > 0.5).float().cpu().numpy())
            labels_spec.append(lbl_spec.cpu().numpy())

    n = len(loader)
    avg_loss = total_loss / n
    
    P_spec = np.vstack(preds_spec)
    L_spec = np.vstack(labels_spec)
    f1_s = f1_score(L_spec, P_spec, average='micro')
    
    return avg_loss, f1_s

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
    
    model = MultiModalNet_NoRegion().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(SAVE_DIR, "best_model.pth")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        t_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        v_loss, v_f1 = eval_epoch(model, val_loader, criterion, DEVICE)    
        
        print(f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        print(f"Val Specific F1: {v_f1:.4f}")
        
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
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
    t_loss, t_f1 = eval_epoch(model, test_loader, criterion, DEVICE)
    print(f"Test Specific F1: {t_f1:.4f}")

if __name__ == "__main__":
    main()