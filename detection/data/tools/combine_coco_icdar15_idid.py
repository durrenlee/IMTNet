import argparse
import json
import os.path
import shutil
import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', help='Root dir for idid icdar15 image files.')
    parser.add_argument('train_json_file', help='idid-coco dataset train json file.')
    parser.add_argument('val_json_file', help='idid-coco dataset val json file.')
    parser.add_argument('labels_dir', help='Root dir for idid-ic15 label txt files.')
    parser.add_argument('out_dir', help='output root dir.')
    # parser.add_argument(
    #     'train_val_ratio', help='train and val ratio. 0<train_val_ratio<1')

    args = parser.parse_args()
    return args

def crop_save(image, points, cropped_out_root, cropped_image_name):
    # Convert points to a numpy array of float32
    src_points = np.array(points, dtype=np.float32)

    # Compute the width and height of the new rectangle
    width1 = np.sqrt((points[1][0] - points[0][0]) ** 2 + (points[1][1] - points[0][1]) ** 2)
    width2 = np.sqrt((points[2][0] - points[3][0]) ** 2 + (points[2][1] - points[3][1]) ** 2)
    maxWidth = max(int(width1), int(width2))

    height1 = np.sqrt((points[3][0] - points[0][0]) ** 2 + (points[3][1] - points[0][1]) ** 2)
    height2 = np.sqrt((points[2][0] - points[1][0]) ** 2 + (points[2][1] - points[1][1]) ** 2)
    maxHeight = max(int(height1), int(height2))

    # Destination points for perspective transformation
    dst_points = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    # Get the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Perform the warp perspective transformation
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Save the cropped image
    cropped_image_path = os.path.join(cropped_out_root, f'{cropped_image_name}')
    cv2.imwrite(cropped_image_path, warped)
    # print(f'cropped_image: {cropped_image_path} finished.')

def _process_impl(json_data, images_dir, labels_dir, out_dir):
    merged_dataset_format = {
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

    img_id = 0
    for img in json_data["images"]:
        # print(img["id"])
        # print(img["file_name"])
        img_file_name = img["file_name"]
        image_path = os.path.join(images_dir, f'{img_file_name}')

        # print(image_path)
        # image data info format
        image_info = {
            "id": img_id,  # reorder
            "file_name": img_file_name,
            "width": img["width"],
            "height": img["height"],
            "obj_annotations": [],
            "label_annotations": []
        }
        # print(image_info)
        #  object annotation of the image
        ann_id = 0  # anno id reorder
        for annotation in json_data["annotations"]:
            if annotation["image_id"] == img["id"]:
                ann_format = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": annotation["category_id"],
                    "bbox": annotation["bbox"],
                    "area": annotation["area"],
                    "iscrowd": annotation["iscrowd"]
                }
                image_info["obj_annotations"].append(ann_format)
                ann_id = ann_id + 1

        label_file_name = img["file_name"] + ".txt"
        label_path = os.path.join(labels_dir, label_file_name)

        # check different extension name
        label_file_exit = False
        if os.path.exists(label_path):
            label_file_exit = True
        else:
            file_name_no_ext = os.path.splitext(img["file_name"])[0]
            label_file_name = file_name_no_ext + ".jpeg.txt"
            label_path = os.path.join(labels_dir, f'{label_file_name}')
            if os.path.exists(label_path):
                label_file_exit = True
            else:
                label_file_name = file_name_no_ext + ".JPG.txt"
                label_path = os.path.join(labels_dir, f'{label_file_name}')
                if os.path.exists(label_path):
                    label_file_exit = True

        if label_file_exit:
            with open(label_path, 'r') as file:
                # read lines once
                lines = file.readlines()

            label_ann_id = 0
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            for line in lines:
                line = line.rstrip('\n')
                # ex: 1135,476,1158,492,1149,503,1126,485,TEST
                # print('--------------------------------')
                # print(line)
                arr = line.split(",")
                label = arr[-1]
                # print(label)
                # TypeError: list indices must be integers or slices, not tuple
                vertex_arr = arr[0:(len(arr) - 1)]
                # print(vertex_arr)
                vertex_int_list = list(map(int, vertex_arr))

                # cropped image name
                # print(img["file_name"])
                file_name_arr = img["file_name"].split(".")
                cropped_img_name = file_name_arr[0:(len(file_name_arr) - 1)]
                cropped_img_name = ''.join(cropped_img_name)
                # print(type(cropped_img_name))
                cropped_img_name = cropped_img_name + "_" + str(label_ann_id) + ".jpg"
                label_ann_format = {
                    "id": label_ann_id,
                    "image_id": img_id,
                    "label": label,
                    "vertex_points": [],
                    "cropped_img": cropped_img_name
                }
                vetex_list = list(zip(vertex_int_list[::2], vertex_int_list[1::2]))
                # print(vetex_list)
                label_ann_format["vertex_points"] = vetex_list

                # crop image region with and save
                cropped_out_root = os.path.join(out_dir, f'cropped_imgs')
                crop_save(image, vetex_list, cropped_out_root, cropped_img_name)

                image_info["label_annotations"].append(label_ann_format)
                label_ann_id = label_ann_id + 1
        else:
            print(f'{label_path} not exist.')

        # copy image to destination
        out_copy_img_path = os.path.join(out_dir, f'images')
        out_copy_img_path = os.path.join(out_copy_img_path, f'{img_file_name}')
        # print(f'source file:{image_path} to des file:{out_copy_img_path}')
        shutil.copy2(image_path, out_copy_img_path)

        merged_dataset_format["data"].append(image_info)
        img_id = img_id + 1

    print('==============finish process implement============================')
    # print(merged_dataset_format)
    return merged_dataset_format


def process(images_dir, train_json_file, val_json_file, labels_dir, out_dir):
    # Load data from idid-coco train json file
    with open(train_json_file, "r") as infile:
        train_json_data = json.load(infile)
    # merge train dataset and copy images
    out_train_dir = os.path.join(out_dir, 'train')
    merged_train_dataset_format = _process_impl(train_json_data, images_dir, labels_dir, out_train_dir)
    # output train json-merged  file
    out_ann_path = os.path.join(out_train_dir, f'annotations')
    out_json_path = os.path.join(out_ann_path, f'annotations.json')
    with open(out_json_path, 'w') as f:
        json.dump(merged_train_dataset_format, f, indent=4)

    # Load data from idid-coco val json file
    with open(val_json_file, "r") as infile:
        val_json_data = json.load(infile)
    # merge val dataset and copy images
    out_val_dir = os.path.join(out_dir, 'val')
    merged_val_dataset_format = _process_impl(val_json_data, images_dir, labels_dir, out_val_dir)
    # output train json-merged  file
    out_ann_val_path = os.path.join(out_val_dir, f'annotations')
    out_val_json_path = os.path.join(out_ann_val_path, f'annotations.json')
    with open(out_val_json_path, 'w') as f:
        json.dump(merged_val_dataset_format, f, indent=4)

def main():
    args = parse_args()
    process(args.images_dir, args.train_json_file, args.val_json_file,
            args.labels_dir, args.out_dir)
    print('finish')


if __name__ == '__main__':
    main()