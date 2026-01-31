import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

class ChestXrayDataset(Dataset):
    def __init__(self, report_csv, label_csv, img_dir):
        self.img_dir = img_dir
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

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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

        label_specific = self.df.iloc[idx][self.specific_classes].values.astype('float32')
        label_specific = torch.tensor(label_specific)
        
        label_region = self.df.iloc[idx][self.region_classes].values.astype('float32')
        label_region = torch.tensor(label_region)

        return image, full_text, label_specific, label_region

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    report_csv = os.path.join(current_dir, '..', 'archive', 'indiana_reports.csv')
    label_csv = os.path.join(current_dir, '..', 'dataset_with_labels_2.csv')
    img_folder = os.path.join(current_dir, '..', 'archive', 'images', 'images_normalized')

    dataset = ChestXrayDataset(report_csv, label_csv, img_folder)
    print(f"Dataset 长度: {len(dataset)}")
        
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