import pandas as pd
import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, '..', 'archive', 'indiana_reports.csv')
save_path = os.path.join(current_dir, '..', 'mesh_problem.json')

df = pd.read_csv(csv_path)

d = {}

for index, row in df.iterrows():
    meshes = str(row['MeSH'])
    problems = str(row['Problems'])
    
    meshes = meshes.split(';')
    problems = problems.split(';')
    for m,p in zip(meshes, problems):
        d[m] = p

data_to_send = []
for mesh, hint in d.items():
    data_to_send.append({
        "original_mesh": mesh,
        "reference_problem": hint
    })

print(len(data_to_send))

with open(save_path, 'w') as f:
    json.dump(data_to_send, f, ensure_ascii=False, indent=4)
