import os
import torch
import pandas as pd
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

class ChestXrayDataset(Dataset):
    def __init__(self, report_csv, label_csv, img_dir, split='train'):
        self.img_dir = img_dir
        self.split = split

        df_reports = pd.read_csv(report_csv)
        df_labels = pd.read_csv(label_csv)
        
        df_reports['findings'] = df_reports['findings'].fillna("No findings.")
        df_reports['impression'] = df_reports['impression'].fillna("No impression.")

        self.df = pd.merge(df_labels, df_reports, on='uid', how='left')
        
        self.df['findings'] = self.df['findings'].fillna("No findings.")
        self.df['impression'] = self.df['impression'].fillna("No impression.")

        self.specific_classes = [
            'Opacity', 'Degenerative Spine Disease', 'Cardiomegaly', 'Medical Device', 
            'Pulmonary Atelectasis', 'Calcinosis', 'Calcified Granuloma', 'Spinal Deformity',
            'Hyperaeration', 'Scar', 'Pleural Effusion', 'Granuloma', 'Emphysema', 
            'Atherosclerosis', 'Lung Nodule', 'Pulmonary Edema', 'Diaphragmatic Elevation', 
            'Fracture', 'Tortuous Aorta', 'Pneumonia', 'Diaphragmatic Flattening', 'Hernia', 
            'Costophrenic Angle Blunting', 'Hyperlucent Lung', 'Hypoinflation', 'Arthritis', 
            'Metabolic Bone Disease', 'Pneumothorax', 'Mediastinal Enlargement', 'Fibrosis'
        ]
        
        self.region_classes = [
            'Lung Pathology', 'Bone & Structure', 'Heart & Mediastinum', 
            'Pleural & Space', 'Foreign Object'
        ]

        self.BANNED_PATTERNS = [
            # 核心病变类 (直接对应 Class Name)
            r"pneumonia",
            r"pneumothorax",
            r"effusion",
            r"atelectasis",
            r"edema",
            r"fibrosis",
            r"emphysema",
            r"hernia",
            r"nodule",
            r"granuloma",
            r"mass",
            r"fracture", 
            r"scar",

            # 视觉特征类 (直接对应 Class Name)
            r"opacity", r"opacities",
            r"hyperaeration",
            r"hyperlucent", r"lucency",
            r"hypoinflation",
            r"calcinosis", r"calcifi",
            r"blunting", r"blunted",
            r"elevation", r"elevated",
            r"flattening", r"flattened",
            r"diaphragm",

            # 骨骼与脊柱类 (直接对应 Class Name)
            r"spondylosis", r"degenerative",
            r"deformity", r"scoliosis", r"kyphosis",
            r"arthrit",
            r"metabolic",

            # 心脏与血管类 (直接对应 Class Name)
            r"cardiomegaly", r"enlarged heart",
            r"mediastin",
            r"atherosclerosis",
            r"tortuous", r"aorta", 

            # 装置与异物类 (增强版 - 包含所有常见器械)
            r"device", r"hardware", r"instrument", r"prosthesis", r"implant",
            r"catheter", r"tube", r"wire", r"lead",
            r"port", 
            r"pacemaker", r"valve", r"clip", r"sternotomy", r"suture",
            r"foreign", r"object",

            # 正常/无恙类
            r"normal", r"clear", r"unremarkable", r"intact", r"stable", r"free",
            r"grossly", r"within limits", r"negative",

            # 程度副词
            r"mild", r"moderate", r"severe", r"acute", r"chronic", r"small", r"large",
            r"bilateral", r"left", r"right", r"upper", r"lower"
        ]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                normalize
            ])

    def sanitize_text(self, text):
        if pd.isna(text) or str(text).strip() == "":
            return "No findings reported."
            
        text = str(text).lower()

        text = re.sub(r'([.,;?!])', r' \1 ', text)
        
        replacement = "[MASK]" 
        
        for pattern in self.BANNED_PATTERNS:
            regex = r'\b' + pattern + r'[a-z]*\b'
            text = re.sub(regex, replacement, text)

        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx]['filename']).strip()
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros(3, 224, 224) 

        findings = str(self.df.iloc[idx]['findings'])
        impression = str(self.df.iloc[idx]['impression'])
        
        full_text = f"Findings: {findings} Impression: {impression}"
        
        clean_text = self.sanitize_text(full_text)

        label_specific = self.df.iloc[idx][self.specific_classes].values.astype('float32')
        label_specific = torch.tensor(label_specific)
        
        label_region = self.df.iloc[idx][self.region_classes].values.astype('float32')
        label_region = torch.tensor(label_region)

        return image, clean_text, label_specific, label_region

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    report_csv = os.path.join(current_dir, '..', 'archive', 'indiana_reports.csv')
    label_csv = os.path.join(current_dir, '..', 'dataset_with_labels_2.csv')
    img_folder = os.path.join(current_dir, '..', 'archive', 'images', 'images_normalized')

    dataset = ChestXrayDataset(report_csv, label_csv, img_folder)
    print(f"Dataset 长度: {len(dataset)}")
        
    for i in range(5):
        _, txt, lbl, _ = dataset[i]
        raw_find = dataset.df.iloc[i]['findings']
        print(f"\n[样本 {i}]")
        print(f"原始: {raw_find[:100]}...")
        print(f"清洗: {txt}")
    
    img, txt, lbl_spec, lbl_reg = dataset[0]
    print(f"Specific Labels 形状: {lbl_spec.shape}")
    print(f"Region Labels 形状: {lbl_reg.shape}")

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
    for batch_imgs, batch_txts, batch_lbl_spec, batch_lbl_reg in loader:
        print(f"Batch 图片形状: {batch_imgs.shape}")
        print(f"Batch 文本数量: {len(batch_txts)}")
        print(f"Batch Specific Labels 形状: {batch_lbl_spec.shape}")
        print(f"Batch Region Labels 形状: {batch_lbl_reg.shape}")
        break