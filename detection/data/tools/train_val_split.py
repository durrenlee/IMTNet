import json
import random
import os
import shutil

# raw json created from labelme
raw_json = "C:\\used_datasets\\idid-coco-str\\idid-coco-str.json"
# new train json
target_train_json = "C:\\used_datasets\\idid-coco-str\\idid-coco-str-train.json"
# new val json
target_val_json = "C:\\used_datasets\\idid-coco-str\\idid-coco-str-val.json"
# source image files
images_source_dir = "C:\\used_datasets\\idid-coco-str\\images"
# train image folder
images_target_train_dir = "C:\\used_datasets\\idid-coco-str\\train\\images"
# val image folder
images_target_val_dir = "C:\\used_datasets\\idid-coco-str\\val\\images"

with open(raw_json, 'r') as f:
    original_data = json.load(f)

random.seed(42)

images = original_data["images"]
# shuffle image objects
random.shuffle(images)

dataset_size = len(images)
print(f'total size:{dataset_size}')
# use 8/2 ratio for split
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
print(f'train_size:{train_size}, val_size:{val_size}')

#  train dataset creation #
raw_annos = original_data["annotations"]
train_set_images = images[:train_size]

coco_format = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 0, "name": "No issues"},
        {"id": 1, "name": "Broken"},
        {"id": 2, "name": "Flashover damage"},
    ],
}

image_id = 0
annotation_id = 1
for train_img in train_set_images:
    # print(train_img)
    # appending image
    image_info = {
        "id": image_id,
        "file_name": train_img["file_name"],
        "width": train_img["width"],
        "height": train_img["height"],
    }
    # print(image_info)
    # copy file to new folder
    temp_source_name = os.path.join(images_source_dir, train_img["file_name"])
    temp_target_name = os.path.join(images_target_train_dir, train_img["file_name"])
    shutil.copy2(temp_source_name, temp_target_name)

    coco_format["images"].append(image_info)

    for ann in raw_annos:
        if ann["image_id"] == train_img["id"]:
            # print(ann)
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": ann["iscrowd"]
            }
            # print(annotation)
            annotation_id = annotation_id + 1
            coco_format["annotations"].append(annotation)

    image_id = image_id + 1

with open(target_train_json, 'w') as f:
    json.dump(coco_format, f, indent=4)
print(f'{target_train_json} is created.')

# clear data for val
coco_format.clear()

# val dataset creation #
coco_format = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 0, "name": "No issues"},
        {"id": 1, "name": "Broken"},
        {"id": 2, "name": "Flashover damage"},
    ],
}
image_id = 0
annotation_id = 1
val_set_images = images[train_size:]
for val_img in val_set_images:
    # print(val_img)
    # appending image
    image_info = {
        "id": image_id,
        "file_name": val_img["file_name"],
        "width": val_img["width"],
        "height": val_img["height"],
    }
    # print(image_info)

    # copy file to new folder
    temp_source_name = os.path.join(images_source_dir, val_img["file_name"])
    temp_target_name = os.path.join(images_target_val_dir, val_img["file_name"])
    shutil.copy2(temp_source_name, temp_target_name)

    coco_format["images"].append(image_info)

    for ann in raw_annos:
        if ann["image_id"] == val_img["id"]:
            # print(ann)
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": ann["iscrowd"]
            }
            # print(annotation)
            annotation_id = annotation_id + 1
            coco_format["annotations"].append(annotation)

    image_id = image_id + 1

with open(target_val_json, 'w') as f:
    json.dump(coco_format, f, indent=4)
print(f'{target_val_json} is created.')
