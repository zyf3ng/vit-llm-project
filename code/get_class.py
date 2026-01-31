import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
labels_csv = os.path.join(current_dir, '..', 'dataset_with_labels.csv')
report_csv = os.path.join(current_dir, '..', 'archive', 'indiana_reports.csv')
save_csv = os.path.join(current_dir, '..', 'dataset_with_labels_2.csv')

client = OpenAI(
    api_key='sk-3d364bbce0ec478d96d0df4c9da162ab',
    base_url="https://api.deepseek.com")

df_reports = pd.read_csv(report_csv)

cols_to_use = ['uid', 'MeSH', 'Problems', 'findings', 'impression']
df_text = df_reports[cols_to_use].fillna('')
df_text['uid'] = df_text['uid'].astype(str)

df_final = pd.read_csv(labels_csv)
df_final['uid'] = df_final['uid'].astype(str)

uid_to_indices = {}
for idx, row in df_final.iterrows():
    uid = row['uid']
    if uid not in uid_to_indices:
        uid_to_indices[uid] = []
    uid_to_indices[uid].append(idx)

target_cols = ["Lung Pathology", "Bone & Structure", "Heart & Mediastinum", "Pleural & Space", "Foreign Object"]
for c in target_cols:
    df_final[c] = 0

def make_context(row):
    parts = []
    parts.append(f"MESH: {row['MeSH']}")
    parts.append(f"PROBLEMS: {row['Problems']}")
    parts.append(f"FINDINGS: {row['findings']}")
    parts.append(f"IMPRESSION: {row['impression']}")
    return "\n".join(parts)


uid_to_labels = {}

def ask_llm(text):
    prompt = """
        你是一个专业的放射科医生。请综合分析输入的报告文本，判断病人是否存在以下 5 大类解剖学区域的异常。

        类别定义：
        1. "Lung Pathology" (肺部病变): 肺炎、肺气肿、结节、肺水肿、纤维化、肺不张、肉芽肿、实变等。
        2. "Bone & Structure" (骨骼): 骨折、脊柱退行性变、脊柱侧弯、关节炎、肋骨异常等。
        3. "Heart & Mediastinum" (心脏与纵隔): 心脏增大、动脉硬化、主动脉迂曲/钙化、纵隔增宽等。
        4. "Pleural & Space" (胸膜与横膈): 胸腔积液、气胸、胸膜增厚、横膈异常、疝气等。
        5. "Foreign Object" (异物): 起搏器、手术夹、导管、术后金属影等。

        注意：
        - 输出严格的 JSON 格式。
        - JSON Key 必须严格是上述纯英文单词,不要带中文翻译，也不要有后缀多余空格。例如"Lung Pathology"不要返回"Lung Pathology (肺部病变)"。
        - Value: 1 (存在) 或 0 (不存在)。
        - 忽略 "XXXX" 脱敏字符。
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": prompt}, 
                      {"role": "user", "content": text}],
            temperature=0.0,
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except:
        return None

match_count = 0
save_interval = 20 # 每处理 20 个病人保存一次文件

for idx, row in tqdm(df_text.iterrows(), total=len(df_text)):
    uid = row['uid']
    text = make_context(row)

    result = ask_llm(text)
 
    if result:
        target_indices = uid_to_indices[uid]
        
        for row_idx in target_indices:
            match_count += 1
            for key in target_cols:
                df_final.at[row_idx, key] = int(result.get(key, 0))
    
    if idx % save_interval == 0:
        df_final.to_csv(save_csv, index=False)

# 保存
df_final.to_csv(save_csv, index=False)
