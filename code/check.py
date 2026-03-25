import pandas as pd

# 替换为你真实的文件路径和ID列名
csv_file_path = "dataset_with_labels_2.csv" 
id_column = "uid"  

specific_labels = ['Opacity', 'Medical Device', 'Calcinosis', 'Hernia', 'Scar'] # 挑几个最容易跨界的测
region_labels = ['Lung Pathology', 'Bone & Structure', 'Heart & Mediastinum', 'Pleural & Space', 'Foreign Object']

# 定义每个小类的“老家”（常理上的主战场）
home_base = {
    'Opacity': 'Lung Pathology',
    'Medical Device': 'Foreign Object',
    'Calcinosis': 'Lung Pathology',
    'Hernia': 'Pleural & Space',
    'Scar': 'Lung Pathology'
}

def hunt_true_anomalies(df):
    print("🚀 开始猎杀【彻底脱轨】的真·跨界幽灵...\n")
    print("="*80)
    
    # 清洗：计算病人患有几种具体疾病，为了严谨，我们还是看那些相对单纯的病人
    df['specific_disease_count'] = df[specific_labels].sum(axis=1)
    
    for spec in specific_labels:
        home = home_base.get(spec)
        if not home or home not in df.columns: continue
            
        # 核心过滤逻辑：
        # 1. 患有该病 (spec == 1)
        # 2. 并且它的老家居然是 0 (home == 0) ！！！
        # 3. 并且疾病总数不能太多（排除全身都是病晚期患者的干扰，比如限制在2种以内）
        out_of_bounds = df[(df[spec] == 1) & (df[home] == 0) & (df['specific_disease_count'] <= 2)]
        
        total_out = len(out_of_bounds)
        if total_out == 0:
            continue
            
        print(f"🚨 【绝对脱轨发现！】: {spec} (老家 '{home}' 居然是 0 的样本有 {total_out} 例)")
        
        for reg in region_labels:
            if reg == home: continue # 不看老家
            
            # 看看它到底跑到哪个大类去了
            reg_samples = out_of_bounds[out_of_bounds[reg] == 1]
            if len(reg_samples) > 0:
                sampled_ids = reg_samples[id_column].tolist()[:3] # 抽几个ID
                print(f"   ┣━ 诡异地出现在了 [{reg}]: {len(reg_samples)} 例 | 🔍 去查这些 ID: {sampled_ids}")
        print("-" * 80)

if __name__ == "__main__":
    try:
        df = pd.read_csv(csv_file_path)
        hunt_true_anomalies(df)
    except FileNotFoundError:
        print(f"❌ 找不到文件，请检查路径。")