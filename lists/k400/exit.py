import os
import pdb
def get_name_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        pairs = [line.rstrip('\n').split(' ', 1) for line in f]   # 保留中间空格
        name   = [p for p in pairs]
    return name

def get_existing_files(data_dir):
    """预先读取所有存在的文件路径到集合中"""
    existing_files = set()
    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                # 获取相对于data_dir的路径
                rel_path = os.path.relpath(os.path.join(root, file), data_dir)
                existing_files.add(rel_path)
    return existing_files

import csv, pathlib

label_csv='/mnt/moonfs/lijia-m2/code02/lijia_MoTE/lists/k400/k400_label.csv'
# 一行读完并反转列
name2id = {row['name'].replace(' ',"_"): int(row['id'])
           for row in csv.DictReader(pathlib.Path(label_csv).read_text().splitlines())}
train_path='/mnt/moonfs/lijia-m2/code02/MoTE/lists/k400/vallist.txt'
out_path='/mnt/moonfs/lijia-m2/code02/lijia_MoTE/lists/k400/vallist_all_exit.txt'
data_dir='/mnt/moonfs/lijia-m2/code02/data/k400/compress/val'

print("正在读取所有存在的文件...")
existing_files = get_existing_files(data_dir)
print(f"找到 {len(existing_files)} 个文件")

train_list=get_name_list(train_path)

# 为每个ID创建计数器，限制每个ID最多200个样本
id_counts = {}
max_samples_per_id = 500

with open(out_path, 'w', encoding='utf-8') as f:
    for item in train_list:
        
        if item[0] in existing_files:
            
            path=item[0]
            name=path.split('/')[0]
            label_id = name2id[name]
            
            # 检查该ID的样本数量是否已达到上限
            if id_counts.get(label_id, 0) < max_samples_per_id:
                id_counts[label_id] = id_counts.get(label_id, 0) + 1
                f.write('val/'+item[0]+' '+str(label_id)+'\n')
        
