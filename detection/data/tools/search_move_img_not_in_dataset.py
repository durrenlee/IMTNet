import json
import random
import os
import shutil

idid_coco_imgs_val_dir = "D:\\Desktop\\图像检索\\IDIDataset\\idid-coco\\val"
idid_coco_val_json = "D:\\Desktop\\图像检索\\IDIDataset\\idid-coco\\labels_v1.2_val_coco.json"
dest_dir = "C:\\used_datasets\\not_in_idid-coco-v1\\val"
with open(idid_coco_val_json, 'r') as f:
    val_data = json.load(f)


for img_name in os.listdir(idid_coco_imgs_val_dir):
    file_exist = False
    for coco_img in val_data["images"]:
        if img_name == coco_img["file_name"]:
            file_exist = True
            break
    if not file_exist:
        print(img_name)
        source_file = os.path.join(idid_coco_imgs_val_dir, img_name)
        dest_file = os.path.join(dest_dir, img_name)
        shutil.move(source_file, dest_file)
