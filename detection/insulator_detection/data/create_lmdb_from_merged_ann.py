import json
import os.path
import lmdb
import cv2
import numpy as np

def parse_merged_json(source_merged_json, cropped_images_path):
    # Load data from the JSON file
    with open(source_merged_json, "r") as infile:
        data = json.load(infile)

    img_label_list = []
    for img_obj in data["data"]:
        # image object has labels
        if len(img_obj["label_annotations"]) > 0:
            for label_obj in img_obj["label_annotations"]:
                image_path = os.path.join(cropped_images_path, label_obj["cropped_img"])
                img_label_format = {
                    "image_id": label_obj["image_id"],
                    "image_path": image_path,
                    "label": label_obj["label"]
                }
                img_label_list.append(img_label_format)

    return img_label_list

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def create_lmdb(img_label_list, des_lmdb_dir, des_map_file):
    nSamples = len(img_label_list)

    image_id_key_dict = {}  # a map: image_id and its list contains cropped image/label keys
    cache = {}
    cnt = 1
    for idx, img_label_obj in enumerate(img_label_list):
        with open(img_label_obj["image_path"], "rb") as f:
            imageBin = f.read()
        try:
            if not checkImageIsValid(imageBin):
                print("%s is not a valid image" % img_label_obj["image_path"])
                continue
        except:
            print("error occurred", idx)
            with open(des_lmdb_dir + "/error_image_log.txt", "a") as log:
                log.write("%s-th image data occurred error\n" % str(idx))
            continue

        '''
        imageKey and labelKey are formatted as strings with zero-padding, followed by encoding to bytes:
        ex imageKey:
        imageKey = "image-%09d".encode() % cnt: when cnt = 1, this creates a key for the image data in the format 
        "image-000000001", where %09d ensures the integer cnt is zero-padded to 9 digits. The .encode() method converts 
        the string to bytes, which is required for LMDB keys.
        '''
        imageKey = "image-%09d".encode() % cnt
        labelKey = "label-%09d".encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = img_label_obj["label"].encode()

        # create the map between image id and index of image/label keys(each index=key-1)
        if str(img_label_obj["image_id"]) in image_id_key_dict:
            image_id_key_dict[str(img_label_obj["image_id"])].append(cnt-1)
        else:
            image_id_key_dict[str(img_label_obj["image_id"])] = []
            image_id_key_dict[str(img_label_obj["image_id"])].append(cnt-1)

        cnt += 1

    # env = lmdb.open(output_path, map_size=int(map_size))
    env = lmdb.open(des_lmdb_dir, map_size=int(1e9))
    cache["num-samples".encode()] = str(nSamples).encode()
    # because the size of the idid marking dataset is small, data is written to lmdb at one time.
    writeCache(env, cache)
    env.close()
    print(f"LMDB dataset created at {des_lmdb_dir} with {nSamples} samples")

    # create image id and keys map
    with open(des_map_file, 'w') as f:
        json.dump(image_id_key_dict, f, indent=4)
    print(f"{des_map_file} created.")


def main():
    source_merged_json = "C:\\used_datasets\\idid-coco-str\\idid-det-str-dataset\\train\\annotations\\annotations.json"
    cropped_images_path = "C:\\used_datasets\\idid-coco-str\\idid-det-str-dataset\\train\\cropped_imgs"
    des_lmdb_dir = "D:\\project\\merge_ds_v2\\lmdb\\val"
    des_map_file = "D:\\project\\merge_ds_v2\\lmdb\\map.json"
    img_label_list = parse_merged_json(source_merged_json, cropped_images_path)
    create_lmdb(img_label_list, des_lmdb_dir, des_map_file)

    with open(des_map_file, "r") as infile:
        map_data = json.load(infile)

    image_id = "27"
    print(map_data[image_id])

if __name__ == '__main__':
    main()