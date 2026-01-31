import json
import os
import time
from openai import OpenAI

API_KEY = "sk-3d364bbce0ec478d96d0df4c9da162ab"
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
# =========================================

def merge(current_tags):
    prompt = f"""
    我有一组医学标签，这是上一轮清洗的结果，但仍然存在同义词没合并干净以及非疾病的词出现的情况。
    
    【任务】：请检查这组标签，并根据如下规则进行合并：
    1.将意思相近或者相同的词合并为同一个标准术语。
    2.如果是单纯的器官，没有其他任何描述的，如"Lung", "Heart", "Aorta", "Thoracic Vertebrae"等，请去除，即映射为null。
    3.如果看到的是"Normal", "No Indexing", "Technical Quality", "Imaged"之类的词，请去除。
    4.如果该词本身已经很精确，就保留本身不做修改。

    【待检查列表】：
    {json.dumps(current_tags)}
    
    【输出格式】：
    JSON字典: {{ "旧标签": "新标签" }} (如果要删除请映射为 null)
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
    input_path = os.path.join(current_dir, '..', 'tag_mapping.json')
    output_path = os.path.join(current_dir, '..', 'tag_mapping_2.json')

    with open(input_path, 'r') as f:
        original_map = json.load(f)

    current_values = list(set([v for v in original_map.values()]))
    print(len(current_values))

    batch_size = 100
    new_map = {}
    
    for i in range(0, len(current_values), batch_size):
        batch = current_values[i : i + batch_size]
        print(f"处理第 {i} 到 {i+len(batch)} 个...", end="")
        
        tag = merge(batch)
        if tag:
            new_map.update(tag)
            print("success")
        else:
            print("failed")

    final_map = {}
    change_count = 0
    
    for key, old_value in original_map.items():    
        if old_value in new_map:
            new_value = new_map[old_value]
            final_map[key] = new_value 
            if new_value != old_value:
                change_count += 1
        else:
            final_map[key] = old_value

    print(change_count)
    
    with open(output_path, 'w') as f:
        json.dump(final_map, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()