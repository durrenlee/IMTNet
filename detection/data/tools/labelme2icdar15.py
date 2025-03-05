
import cv2
import numpy as np
import os
import os.path as osp
import argparse
import glob
import json
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_dir', help='Root dir for labelme json file.')
    parser.add_argument('image_dir', help='Root dir for image file.')
    parser.add_argument(
        'out_dir', help='Dir to save annotations in icdar2015 format.')
    parser.add_argument(
        'task',
        default='standard',
        help='crop or standard, crop for cropping text region into small image, standard for standard icdar 2015')

    args = parser.parse_args()
    return args

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def crop_and_save(cropped_out_root, ann_out_root, image, points, label, total_label):
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

    cropped_image_path = os.path.join(cropped_out_root, f'img_{total_label}.jpg')
    print('cropped_image_path:')

    print(cropped_image_path)
    cv2.imwrite(cropped_image_path, warped)

    # Save the annotation
    annotation_path = os.path.join(ann_out_root, f'gt_img_{total_label}.txt')
    with open(annotation_path, 'w') as f:
        # Write the vertices and label
        vertices_str = ",".join([f"{int(x)},{int(y)}" for x, y in points])
        f.write(f"{vertices_str},{label}")

    print(f'Saved cropped image: {cropped_image_path} and annotation: {annotation_path}')

def copy_and_save(shapes_data, image_path, new_images_dir, new_ann_dir, file_name):
    label_list = []
    points_list = []
    for i, shape in enumerate(shapes_data):
        label_list.append(shape['label'])
        points_list.append(shape['points'])

    # Save the annotation
    # annotation_path = os.path.join(new_ann_dir, f'gt_img_{json_file_no}.txt')
    annotation_path = os.path.join(new_ann_dir, f'{file_name}.txt')
    with open(annotation_path, 'w') as f:
        # Write the vertices and label
        for i in range(len(label_list)):
            vertices_str = ",".join([f"{int(x)},{int(y)}" for x, y in points_list[i]])
            if i == len(label_list) - 1:
                f.write(f"{vertices_str},{label_list[i]}")
            else:
                f.write(f"{vertices_str},{label_list[i]}\n")

    # copy source image to destination
    # to_image_path = os.path.join(new_images_dir, f'img_{json_file_no}.jpg')
    to_image_path = os.path.join(new_images_dir, f'{file_name}')
    shutil.copy2(image_path, to_image_path)

    print(f'destination image: {to_image_path} and annotation: {annotation_path}')

def process(json_dir, img_dir, out_dir, task):
    mkdir_or_exist(out_dir)
    new_images_dir = os.path.join(out_dir, "images")
    mkdir_or_exist(new_images_dir)
    new_ann_dir = os.path.join(out_dir, "annotations")
    mkdir_or_exist(new_ann_dir)
    json_file_list = glob.glob(osp.join(json_dir, '*.json'))

    json_file_no = 1  # the order of icdar15 starts at 1
    total_label = 0   # total label number
    for a_json_file in json_file_list:
        print(a_json_file)
        with open(a_json_file, "r") as infile:
            data = json.load(infile)
            file_name = data["imagePath"].split("\\")[2]
            image_path = img_dir + "\\" + file_name
            print(image_path)
            # image = cv2.imread(image_path)
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)

            # Create output directories for cropped images and annotation files
            if task == 'crop':
                for i, shape in enumerate(data['shapes']):
                    label = shape['label']
                    points = shape['points']
                    crop_and_save(new_images_dir, new_ann_dir, image, points, label, total_label)
                    total_label = total_label + 1
            else:
                # copy_and_save(data['shapes'], image_path, new_images_dir, new_ann_dir, json_file_no)
                copy_and_save(data['shapes'], image_path, new_images_dir, new_ann_dir, file_name)

        json_file_no = json_file_no + 1

def main():
    args = parse_args()

    process(args.json_dir, args.image_dir, args.out_dir, args.task)

    print('finish')


if __name__ == '__main__':
    main()