import json
import os
import os.path as osp
import shutil

# all image files in idid-str dataset
idid_str_images_dir = "D:\project\带型号标记的缺陷绝缘子\labelme\img_data"
image_files = os.listdir(idid_str_images_dir)
assert len(image_files) == 830

# Load idid-coco train data from the JSON file
idid_coco_train_json = "C:\\used_datasets\\combined-idid-coco\\labels_v1.2_train_coco.json"
with open(idid_coco_train_json, "r") as train_infile:
    idid_coco_train_data = json.load(train_infile)
print(f'idid_coco_train_data length:{len(idid_coco_train_data)}')

# Load idid-coco val data from the JSON file
idid_coco_val_json = "C:\\used_datasets\\combined-idid-coco\\labels_v1.2_val_coco.json"
with open(idid_coco_val_json, "r") as val_infile:
    idid_coco_val_data = json.load(val_infile)
print(f'idid_coco_val_data length:{len(idid_coco_val_data)}')

# load additional idid-coco data from json file
idid_coco_additional_json = "C:\\used_datasets\\idid_v1.2_additional\\annotations.json"
with open(idid_coco_additional_json, "r") as addi_infile:
    idid_coco_addi_data = json.load(addi_infile)
print(f'idid_coco_addi_data length:{len(idid_coco_addi_data)}')

target_dir = "C:\\used_datasets\\idid-coco-str\\images"
addi_dir = "C:\\used_datasets\\idid_v1.2_additional\\JPEGImages"

coco_format = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 0, "name": "No issues"},
        {"id": 1, "name": "Broken"},
        {"id": 2, "name": "Flashover damage"},
    ],
}

addi_count = 0
train_count = 0
val_count = 0

image_id = 0
annotation_id = 1
for img in image_files:
    print(f"target img:{img}")
    file_name = img
    width = 0
    height = 0

    found_flag = False
    # search in additional anno file by 1st order
    for addi_img in idid_coco_addi_data["images"]:
        base = osp.splitext(osp.basename(img))[0]
        format_img = base + '.jpg'
        if addi_img['file_name'][11:] == format_img:
            file_name = format_img
            width = addi_img['width']
            height = addi_img['height']
            # appending annotations
            for addi_anno in idid_coco_addi_data["annotations"]:
                if addi_anno["image_id"] == addi_img["id"]:
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": addi_anno["category_id"] - 1,
                        "bbox": addi_anno["bbox"],
                        "area": addi_anno["area"],
                        "iscrowd": addi_anno["iscrowd"]
                    }
                    # print(f'ann id:{addi_anno["id"]}')
                    # print(annotation)
                    annotation_id = annotation_id + 1
                    coco_format["annotations"].append(annotation)
            # copy file to new folder
            temp_source_name = os.path.join(addi_dir, file_name)
            temp_target_name = os.path.join(target_dir, file_name)
            shutil.copy2(temp_source_name, temp_target_name)
            addi_count = addi_count + 1
            found_flag = True
            break

    # search in idid-coco train anno file by 2nd order
    if found_flag is False:
        for train_img in idid_coco_train_data["images"]:
            if train_img["file_name"] == img:
                # print(f'{img} in train data')
                file_name = img
                width = train_img['width']
                height = train_img['height']
                # appending annotations
                for train_anno in idid_coco_train_data["annotations"]:
                    if train_anno["image_id"] == train_img["id"]:
                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": train_anno["category_id"] - 1,
                            "bbox": train_anno["bbox"],
                            "area": train_anno["area"],
                            "iscrowd": train_anno["iscrowd"]
                        }
                        # print(f'ann id:{train_anno["id"]}')
                        # print(annotation)
                        annotation_id = annotation_id + 1
                        coco_format["annotations"].append(annotation)
                # copy file to new folder
                temp_source_name = os.path.join(idid_str_images_dir, img)
                temp_target_name = os.path.join(target_dir, img)
                shutil.copy2(temp_source_name, temp_target_name)
                train_count = train_count + 1
                found_flag = True
                break

    # search in idid-coco val anno file by 3rd order
    if found_flag is False:
        for val_img in idid_coco_val_data["images"]:
            if val_img["file_name"] == img:
                # print(f'{img} in val data')
                file_name = img
                width = val_img['width']
                height = val_img['height']
                # appending annotations
                for val_anno in idid_coco_val_data["annotations"]:
                    if val_anno["image_id"] == val_img["id"]:
                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": val_anno["category_id"] - 1,
                            "bbox": val_anno["bbox"],
                            "area": val_anno["area"],
                            "iscrowd": val_anno["iscrowd"]
                        }
                        # print(f'ann id:{val_anno["id"]}')
                        # print(annotation)
                        annotation_id = annotation_id + 1
                        coco_format["annotations"].append(annotation)
                # copy file to new folder
                temp_source_name = os.path.join(idid_str_images_dir, img)
                temp_target_name = os.path.join(target_dir, img)
                shutil.copy2(temp_source_name, temp_target_name)
                val_count = val_count + 1
                # found_flag = True
                break

    # appending image
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
    }
    # print(image_info)
    coco_format["images"].append(image_info)
    image_id = image_id + 1

# Save the COCO format data to a new JSON file
new_json = "C:\\used_datasets\\idid-coco-str\\idid-coco-str.json"
with open(new_json, "w") as outfile:
    json.dump(coco_format, outfile, indent=4)

assert len(coco_format["images"]) == len(image_files)
print(f'addi_count:{addi_count}, train_count:{train_count}, val_count:{val_count}')
assert (addi_count + train_count + val_count) == len(image_files)
print(f'{len(image_files)} files are processed successfully.')
