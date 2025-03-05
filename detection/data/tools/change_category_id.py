import json
import os

source_json_dir = "D:\\Desktop\\图像检索\\IDIDataset\\idid-coco"
source_json_name = "labels_v1.2_val_coco.json"
source_file_path = os.path.join(source_json_dir, source_json_name)
with open(source_file_path, 'r') as f:
    original_data = json.load(f)

coco_format = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 0, "name": "No issues"},
        {"id": 1, "name": "Broken"},
        {"id": 2, "name": "Flashover damage"},
    ],
}

# images
image_id = 0
annotation_id = 1
for img_obj in original_data["images"]:
    # change image id
    image_info = {
        "id": image_id,
        "file_name": img_obj["file_name"],
        "width": img_obj["width"],
        "height": img_obj["height"],
    }
    coco_format["images"].append(image_info)

    #  annotations
    ann_exist = False
    for ann_obj in original_data["annotations"]:
        if ann_obj["image_id"] == img_obj["id"]:
            annotation = {
                "id": annotation_id,
                "image_id": image_id,  # replace by new image id
                "category_id": ann_obj["category_id"] - 1,  # reduce 1
                "bbox": ann_obj["bbox"],
                "area": ann_obj["area"],
                "iscrowd": ann_obj["iscrowd"]
            }
            annotation_id = annotation_id + 1
            ann_exist = True
            coco_format["annotations"].append(annotation)
    if not ann_exist:
        print(img_obj)
    image_id = image_id + 1

# save to new file
new_json = os.path.join(source_json_dir, "labels_v1.2_val_coco_new.json")
with open(new_json, 'w') as f:
    json.dump(coco_format, f, indent=4)
print(f'{new_json} is created.')
