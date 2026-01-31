import json
import os
from openai import OpenAI

API_KEY = "sk-3d364bbce0ec478d96d0df4c9da162ab" 
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
BATCH_SIZE = 50 

def clean_batch(batch_data):
    prompt = f"""
    我需要清洗医学标签。
    我给你提供一组数据，包含 "original_mesh" (原始标签) 和 "reference_problem" (参考类别)。
    请输出 "original_mesh" 对应的最精准、最核心的视觉病理名词。
    不要那种很广泛的名称，尽量具体。
    请参考reference_problem提供的标签。
    输出格式每个单词都统一为首字母大写，后面小写。

    要处理的数据是：
    {json.dumps(batch_data)}
    
    输出格式：
    JSON字典: {{ "原始标签名": "清洗后的标签名" }}
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={'type': 'json_object'}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"error: {e}")
        return {}

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, '..', 'mesh_problem.json')
    output_path = os.path.join(current_dir, '..', 'tag_mapping.json')

    with open(input_path, 'r') as f:
        data = json.load(f)
    
    print(f"加载了 {len(data)} 个数据")
    
    all_mappings = {}
    
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i : i + BATCH_SIZE]
        print(f"正在处理第 {i} - {i+len(batch)} 个...", end="")
        
        cleaned = clean_batch(batch)
        
        if cleaned:
            all_mappings.update(cleaned)
            print("success")
        else:
            print("failed")

    with open(output_path, 'w') as f:
        json.dump(all_mappings, f, ensure_ascii=False, indent=4)
    
    print("completed")

if __name__ == "__main__":
    main()