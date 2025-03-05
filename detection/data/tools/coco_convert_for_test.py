import argparse
import json
import os.path
import shutil
import numpy as np
import cv2

'''
create combined det and str from standard idid-coco dataset for testing detection solution
label_annotation is dummy
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', help='idid-coco train(val) json file.')
    parser.add_argument('out_combined_json_file', help='output combined json file.')
    parser.add_argument('dummy_cropped_img', help='dummy_cropped_img.')
    parser.add_argument('copied_dummy_cropped_img_dir', help='copied_dummy_cropped_img_dir.')

    args = parser.parse_args()
    return args

def process(json_file, out_combined_json_file, dummy_cropped_img, copied_dummy_cropped_img_dir):
    # Load data from idid-coco json file
    with open(json_file, "r") as infile:
        train_json_data = json.load(infile)

    # create combined json file
    combined_json_format = {
        "data": [],
        "categories": [
                {
                    "id": 0,
                    "name": "No issues"
                },
                {
                    "id": 1,
                    "name": "Broken"
                },
                {
                    "id": 2,
                    "name": "Flashover damage"
                }
            ]
    }

    for img in train_json_data["images"]:
        # image data info format
        image_info = {
            "id": img["id"],
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "obj_annotations": [],
            "label_annotations": []
        }
        #  object annotation of the image
        ann_id = 0   # anno id reorder
        for annotation in train_json_data["annotations"]:
            if annotation["image_id"] == img["id"]:
                ann_format = {
                    "id": ann_id,
                    "image_id": img["id"],
                    "category_id": annotation["category_id"] - 1,
                    "bbox": annotation["bbox"],
                    "area": annotation["area"],
                    "iscrowd": annotation["iscrowd"]
                }
                image_info["obj_annotations"].append(ann_format)
                ann_id = ann_id + 1

        label_ann_id = 0
        # cropped image name
        file_name_arr = img["file_name"].split(".")
        cropped_img_name = file_name_arr[0:(len(file_name_arr) - 1)]
        cropped_img_name = ''.join(cropped_img_name)
        cropped_img_name = cropped_img_name + "_" + str(label_ann_id) + ".jpg"
        label_ann_format = {
            "id": label_ann_id,
            "image_id": img["id"],
            "label": 'TEST',
            "vertex_points": [
                [
                    3286,
                    1342
                ],
                [
                    3328,
                    1322
                ],
                [
                    3370,
                    1418
                ],
                [
                    3328,
                    1435
                ]
            ],
            "cropped_img": cropped_img_name
        }

        copied_dummy_cropped_img_path = os.path.join(copied_dummy_cropped_img_dir, cropped_img_name)
        shutil.copy2(dummy_cropped_img, copied_dummy_cropped_img_path)
        image_info["label_annotations"].append(label_ann_format)

        combined_json_format["data"].append(image_info)
    # print(merged_dataset_format)


    # output train json-merged  file
    with open(out_combined_json_file, 'w') as f:
        json.dump(combined_json_format, f, indent=4)


def main():
    args = parse_args()
    process(args.json_file, args.out_combined_json_file, args.dummy_cropped_img,
            args.copied_dummy_cropped_img_dir)
    print('finish')

if __name__ == '__main__':
    main()