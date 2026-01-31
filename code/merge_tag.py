import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_dir, '..', 'tag_mapping_2.json') 
save_path = os.path.join(current_dir, '..', 'tag_mapping_3.json')

with open(input_path, 'r') as f:
    mapping_dict = json.load(f)


merge_rules = {
    # 1. Opacity (混浊/阴影)
    "Lung Opacity": "Opacity",
    "Airspace Disease": "Opacity",
    "Infiltrate": "Opacity",
    "Consolidation": "Opacity",
    "Lung Density": "Opacity",
    "Interstitial Opacity": "Opacity",
    "Diffuse Interstitial Markings": "Opacity",
    "Pulmonary Opacity": "Opacity",
    "Thoracic Opacity": "Opacity",
    "Hilar Opacity": "Opacity",
    "Alveolar Opacity": "Opacity",
    "Cardiophrenic Angle Density": "Opacity",
    "Supracardiac Opacity": "Opacity",
    "Left Apical Opacity": "Opacity",
    "Right Basal Airspace Disease": "Opacity",
    "Lingular Opacity": "Opacity",
    "Patchy Lung Opacity": "Opacity",
    "Right Basal Opacity": "Opacity",
    "Costophrenic Angle Opacities": "Opacity",
    "Paratracheal Density": "Opacity",
    "Retrocardiac Density": "Opacity",
    "Opacity Lung Streaky": "Opacity",
    "Thoracic Density": "Opacity",
    "Upper Lobe Density": "Opacity",
    "Bilateral Lung Base Opacities": "Opacity",
    "Lung Airspace Disease": "Opacity",

    # 2. Atelectasis (肺不张)
    "Bilateral Lung Atelectasis": "Pulmonary Atelectasis",
    "Atelectasis": "Pulmonary Atelectasis",
    "Pulmonary Atelectasis Apex Right": "Pulmonary Atelectasis",
    "Pulmonary Atelectasis Base Left Streaky Mild": "Pulmonary Atelectasis",

    # 3. Cardiomegaly (心脏肥大)
    "Cardiac Shadow Enlargement": "Cardiomegaly",
    "Left Ventricular Enlargement": "Cardiomegaly",
    "Atrial Enlargement": "Cardiomegaly",
    "Atrial Prominence": "Cardiomegaly",
    "Cardiac Shadow Irregularity": "Cardiomegaly",

    # 4. Calcification & Granuloma (钙化与肉芽肿)
    # 分两类：Granuloma (肉芽肿) 和 Calcinosis (单纯钙化)
    "Multiple Calcified Granulomas": "Calcified Granuloma",
    "Hilar Calcified Granuloma": "Calcified Granuloma",
    "Thoracic Calcified Granuloma": "Calcified Granuloma",
    "Paratracheal Calcified Granuloma": "Calcified Granuloma",
    
    "Granulomatous Disease": "Granuloma",
    "Lung Granulomas": "Granuloma",
    "Granulomas": "Granuloma",

    "Lung Calcinosis": "Calcinosis",
    "Hilar Calcinosis": "Calcinosis",
    "Hilar Calcification": "Calcinosis",
    "Lymph Node Calcinosis": "Calcinosis",
    "Bilateral Hilar Calcinosis": "Calcinosis",
    "Tracheal Calcinosis": "Calcinosis",
    "Rib Calcinosis": "Calcinosis",
    "Abdominal Calcification": "Calcinosis",
    "Pleural Calcinosis": "Calcinosis",
    "Pulmonary Calcification": "Calcinosis",
    "Thoracic Calcifications": "Calcinosis",
    "Mediastinal Calcinosis": "Calcinosis",
    "Aortic Calcification": "Calcinosis",
    "Coronary Calcification": "Calcinosis",
    "Calcification": "Calcinosis",
    "Bilateral Lung Lymph Node Calcinosis": "Calcinosis",

    # 5. Nodule (结节)
    "Lung Nodules": "Lung Nodule",
    "Pulmonary Nodules": "Lung Nodule",
    "Thoracic Nodules": "Lung Nodule",
    "Bilateral Lower Lobe Nodules": "Lung Nodule",

    # 6. Spine & Bone (脊柱与骨骼)
    # 合并为三大类: Degenerative Spine (退行), Spinal Deformity (畸形), Fracture (骨折)
    "Vertebral Deformity": "Spinal Deformity", 
    "Kyphosis": "Spinal Deformity",
    "Scoliosis": "Spinal Deformity",
    "Thoracic Scoliosis": "Spinal Deformity",
    "Thoracic Vertebral Deformity": "Spinal Deformity",
    "Kyphosis Thoracic Vertebrae Mild": "Spinal Deformity",
    "Spinal Fusion": "Spinal Deformity", 

    "Degenerative Disease": "Degenerative Spine Disease",
    "Thoracic Spondylosis": "Degenerative Spine Disease",
    "Thoracic Osteophytes": "Degenerative Spine Disease",
    "Spondylosis": "Degenerative Spine Disease",
    "Degenerative Changes": "Degenerative Spine Disease",
    "Spinal Degeneration": "Degenerative Spine Disease",
    "Lumbar Vertebrae Degeneration": "Degenerative Spine Disease",
    "Degenerative Spine": "Degenerative Spine Disease",
    "Spinal Osteophyte": "Degenerative Spine Disease",
    "Osteophytes": "Degenerative Spine Disease",
    "Cervical Osteophytes": "Degenerative Spine Disease",

    "Bone Fractures": "Fracture",
    "Rib Fractures": "Fracture",
    "Clavicle Fracture": "Fracture",
    "Humerus Fracture": "Fracture",
    "Vertebral Fractures": "Fracture",

    # 7. Emphysema & COPD (肺气肿)
    "Chronic Obstructive Pulmonary Disease": "Emphysema", 
    "Pulmonary Emphysema": "Emphysema",
    "Bullous Emphysema": "Emphysema",
    "Emphysema Severe": "Emphysema",
    
    # 8. Hyperdistention (肺过度充气 - 也是COPD的一种表现)
    "Hyperdistention": "Hyperaeration", 
    "Lung Hyperdistention": "Hyperaeration",
    "Lung Bilateral Hyperdistention Mild": "Hyperaeration",
    "Hypoinflation": "Hypoinflation", # 反义词，保留

    # 9. Pleural Effusion (胸腔积液)
    "Right Pleural Effusion": "Pleural Effusion",
    "Left Pleural Effusion": "Pleural Effusion",
    "Hydropneumothorax": "Pleural Effusion", # 包含积液

    # 10. Scar & Fibrosis (瘢痕与纤维化)
    "Lung Cicatrix": "Scar",
    "Cicatrix": "Scar",
    "Apical Scarring": "Scar",
    "Costophrenic Cicatrix": "Scar",
    "Lung Scar": "Scar",
    
    "Pulmonary Fibrosis": "Fibrosis",
    "Lung Fibrosis": "Fibrosis",
    "Bilateral Lower Lobe Pulmonary Fibrosis": "Fibrosis",

    # 11. Diaphragm (膈肌)
    "Elevated Diaphragm": "Diaphragmatic Elevation",
    "Flattened Diaphragm": "Diaphragmatic Flattening",
    "Diaphragmatic Eventration": "Diaphragmatic Elevation",

    # 12. Aorta (主动脉)
    "Thoracic Aortic Tortuosity": "Tortuous Aorta",
    "Aortic Tortuosity": "Tortuous Aorta",

    # 13. Other (其他)
    "Foreign Body": "Medical Device",
    "Shoulder Implant": "Medical Device",
    "Shoulder Foreign Body": "Medical Device",
    "Hiatal Hernia": "Hernia",
    "Hernia, Diaphragmatic": "Hernia",
    "Pulmonary Congestion": "Pulmonary Edema", 
    "Pulmonary Edema": "Pulmonary Edema",
    "Bilateral Basal Pneumonia": "Pneumonia",
    "Right Lower Lobe Pneumonia": "Pneumonia",
    "Bone Diseases, Metabolic": "Metabolic Bone Disease",

    # === 补丁区 (Patch) ===
    "Blunted Costophrenic Angle": "Costophrenic Angle Blunting",
    "Lung Lucency": "Hyperlucent Lung",
    "Mediastinal Prominence": "Mediastinal Enlargement",
    "Pulmonary Consolidation": "Opacity",
    "Retrocardiac Opacity": "Opacity",
    "Left Lower Lobe Opacity": "Opacity",
    "Bilateral Lower Lobe Opacity": "Opacity",
    "Subcutaneous Emphysema": "Emphysema", # 皮下气肿也是气肿，或者是单独一类，量少建议合或者不管
    "Deformity": "Spinal Deformity", # 这里的 Deformity 通常指骨骼
    "Sclerosis": "Degenerative Spine Disease" # 骨硬化通常伴随退行性变
}

count = 0
for key, value in mapping_dict.items():
    if value in merge_rules:
        mapping_dict[key] = merge_rules[value]
        count += 1

print(f"修正了 {count} 个映射")

device_keywords = [
    "Catheter",    # 导管
    "Tube",        # 插管
    "Implant",     # 植入物
    "Device",      # 设备
    "Pacemaker",   # 起搏器
    "Wire",        # 钢丝
    "Suture",      # 缝线
    "Clip",        # 手术夹
    "Hardware",    # 五金/固定件
    "Sternotomy",  # 胸骨切开术(通常意味着有钢丝)
    "Valve",       # 人工瓣膜
    "Port-A-Cath",  # 输液港
    "Instrument",  # 器械
    "Prosthesis",  # 假体
]

count_device = 0

for key in mapping_dict.keys():
    for kw in device_keywords:
        if kw.lower() in key.lower():
            mapping_dict[key] = "Medical Device"
            count_device += 1
            break

print(f"归类了 {count_device} 个医疗器械标签")

with open(save_path, 'w') as f:
    json.dump(mapping_dict, f, ensure_ascii=False, indent=4)
