import os
import json
import shutil


detstr_coco_json = "C:\\used_datasets\\idid-coco-str\\idid-det-str-dataset\\train\\annotations\\annotations_appended.json"
with open(detstr_coco_json, 'r') as f:
    data = json.load(f)

for img_obj in data["data"]:
    print(img_obj["file_name"])