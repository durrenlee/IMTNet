import os
import json
import shutil

# defect ds train images
idid_coco_imgs_dir = "D:\\Desktop\\图像检索\\IDIDataset\\idid-coco\\val"
target_detstr_img_dir = "C:\\used_datasets\\idid-coco-str\\idid-det-str-dataset\\val\\images"

detstr_coco_json = "C:\\used_datasets\\idid-coco-str\\idid-det-str-dataset\\val\\annotations\\annotations.json"
with open(detstr_coco_json, 'r') as f:
    data = json.load(f)

last_id = len(data["data"]) - 1
print(f"det-str current id:{last_id}")

# idid coco annotation json
idid_coco_json = "D:\\Desktop\\图像检索\\IDIDataset\\idid-coco\\labels_v1.2_val_coco.json"
with open(idid_coco_json, 'r') as f:
    idid_coco_json_data = json.load(f)

for img_name in os.listdir(idid_coco_imgs_dir):
    # check if image exists in current det-str combined dataset
    img_exist = False
    for detstr_obj in data["data"]:
        if detstr_obj["file_name"].upper() == img_name.upper():
            img_exist = True
            break

    # append to detstr dataset annotation
    if not img_exist:
        print(f"img_name:{img_name}")
        last_id = last_id + 1
        for idid_img_obj in idid_coco_json_data["images"]:
            if idid_img_obj["file_name"].upper() == img_name.upper():
                # print(idid_img_obj)
                image_info = {
                    "id": last_id,
                    "file_name": img_name,
                    "width": idid_img_obj["width"],
                    "height": idid_img_obj["height"],
                    "obj_annotations": [],
                    "label_annotations": []
                }

                #  object annotation of the image
                ann_id = 0  # anno id reorder
                for annotation in idid_coco_json_data["annotations"]:
                    if annotation["image_id"] == idid_img_obj["id"]:
                        ann_format = {
                            "id": ann_id,
                            "image_id": last_id,
                            "category_id": annotation["category_id"],
                            "bbox": annotation["bbox"],
                            "area": annotation["area"],
                            "iscrowd": annotation["iscrowd"]
                        }
                        image_info["obj_annotations"].append(ann_format)
                        ann_id = ann_id + 1
                # print('----------------------------')
                # print(image_info)
                image_path = os.path.join(idid_coco_imgs_dir, img_name)
                to_image_path = os.path.join(target_detstr_img_dir, img_name)
                shutil.copy2(image_path, to_image_path)
                data["data"].append(image_info)

print(f'updated file:{len(data["data"])}')

new_json = "C:\\used_datasets\\idid-coco-str\\idid-det-str-dataset\\val\\annotations\\annotations_appended.json"
with open(new_json, 'w') as f:
    json.dump(data, f, indent=4)
print(f'{new_json}, append completed.')