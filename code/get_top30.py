import pandas as pd
import json
import os
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, '..', 'archive', 'indiana_reports.csv')
mapping_path = os.path.join(current_dir,'..', 'tag_mapping_3.json')
save_path = os.path.join(current_dir, '..', 'top30_tag.json')


df = pd.read_csv(csv_path)
with open(mapping_path, 'r') as f:
    mapping_dict = json.load(f)

counter = Counter()

for meshes in df['MeSH']:
    tags = [t.strip() for t in meshes.split(';') if t.strip()]
    for t in tags:
        if t in mapping_dict:
            new_tag = mapping_dict[t]
            if new_tag:
                counter[new_tag] += 1

top_30_tuples = counter.most_common(30)
top_30_list = [t[0] for t in top_30_tuples]

for i, (tag, count) in enumerate(top_30_tuples):
    print(i+1, tag, count)

with open(save_path, 'w') as f:
    json.dump(top_30_list, f, indent=4)
