import os
import shutil
import json

# 定义源目录和目标目录
source_dir = 'D:\\idid_data\\idid\\images'
train_dir = 'D:\\idid_data\idid_coco\\train'
val_dir = 'D:\\idid_data\\idid_coco\\val'

# 创建目标目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 读取训练集文件
with open('../jsons/labels_v1.2_train.json', 'r') as f:
    train_data = json.load(f)

# 复制训练集中的图像到train目录
for item in train_data:
    filename = item['filename']
    source_path = os.path.join(source_dir, filename)
    target_path = os.path.join(train_dir, filename)
    shutil.copy(source_path, target_path)

# 读取验证集文件
with open('../jsons/labels_v1.2_val.json', 'r') as f:
    val_data = json.load(f)

# 复制验证集中的图像到val目录
for item in val_data:
    filename = item['filename']
    source_path = os.path.join(source_dir, filename)
    target_path = os.path.join(val_dir, filename)
    shutil.copy(source_path, target_path)

print("Files have been copied successfully.")
