import pandas as pd
import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
report_file = os.path.join(current_dir, '..', 'archive', 'indiana_reports.csv')
projections_file = os.path.join(current_dir, '..', 'archive', 'indiana_projections.csv')
top30_file = os.path.join(current_dir, '..', 'top30_tag.json')
mapping_file = os.path.join(current_dir, '..', 'tag_mapping_3.json')
save_csv = os.path.join(current_dir, '..', 'dataset_with_labels.csv')

df_reports = pd.read_csv(report_file).fillna('')
df_projections = pd.read_csv(projections_file).fillna('')

with open(mapping_file, 'r') as f:
    raw_dict = json.load(f)

mapping_dict = {k.strip(): v for k, v in raw_dict.items()}

with open(top30_file, 'r') as f:
    labels = json.load(f)
    
uid_to_labels = {}

for idx, row in df_reports.iterrows():
    uid = row['uid']
    meshes = row['MeSH']
    current_tags = set()
    
    if meshes:
        mesh = [m.strip() for m in meshes.split(';')]
        for m in mesh:
            tag = mapping_dict[m]
            if tag:
                current_tags.add(tag)
    
    uid_to_labels[uid] = current_tags 

final_data = []

for idx, row in df_projections.iterrows():
    uid = row['uid']
    filename = row['filename']
    projection = row['projection']
    
    patient_tags = uid_to_labels.get(uid, set())
    
    patient_dict = {
        'uid': uid,
        'image_id': filename,
        'projection': projection
    }
    
    for label in labels:
        if label in patient_tags:
            patient_dict[label] = 1
        else:
            patient_dict[label] = 0
            
    final_data.append(patient_dict)

df_final = pd.DataFrame(final_data)

cols = ['uid', 'image_id', 'projection'] + labels
df_final = df_final[cols]

df_final.to_csv(save_csv, index=False)