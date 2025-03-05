'''
lmdb dataset is simplify created following the steps from
https://github.com/ku21fan/STR-Fewer-Labels/blob/main/create_lmdb_dataset.py
'''
import os
import io
import argparse
import lmdb
import cv2
import numpy as np
import pickle
import random
import unicodedata
from PIL import Image
import re

class CharsetAdapter:
    """Transforms labels according to the target charset."""

    def __init__(self, target_charset) -> None:
        super().__init__()
        self.lowercase_only = target_charset == target_charset.lower()
        self.uppercase_only = target_charset == target_charset.upper()
        self.unsupported = f'[^{re.escape(target_charset)}]'

    def __call__(self, label):
        if self.lowercase_only:
            label = label.lower()
        elif self.uppercase_only:
            label = label.upper()
        # Remove unsupported characters
        label = re.sub(self.unsupported, '', label)
        return label

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def create_lmdb_dataset(output_path, image_label_list, train_val_ratio=0.8, map_size=1e7):
    """
    Create an LMDB dataset for text recognition.

    Args:
        output_path (str): Path to the output LMDB database.
        image_label_list (list of tuples): List of (image_path, label) pairs.
        train_val_ratio(float): Train and val ratio
        map_size (int, optional): Maximum size of the database in bytes. Default is 1TB.
    """
    train_size = int(len(image_label_list) * float(train_val_ratio))
    train_output_path = os.path.join(output_path, 'train')
    val_output_path = os.path.join(output_path, 'val')
    create_a_dataset(train_output_path, image_label_list[0:train_size])
    create_a_dataset(val_output_path, image_label_list[train_size:])


def create_a_dataset(output_path, image_label_sub_list):
    os.makedirs(output_path, exist_ok=True)
    nSamples = len(image_label_sub_list)
    cache = {}
    cnt = 1

    # Iterate over each image-label pair
    for idx, (image_path, label) in enumerate(image_label_sub_list):
        with open(image_path, "rb") as f:
            imageBin = f.read()
        try:
            if not checkImageIsValid(imageBin):
                print("%s is not a valid image" % image_path)
                continue
        except:
            print("error occurred", idx)
            with open(output_path + "/error_image_log.txt", "a") as log:
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
        cache[labelKey] = label.encode()

        cnt += 1

    # env = lmdb.open(output_path, map_size=int(map_size))
    env = lmdb.open(output_path)
    cache["num-samples".encode()] = str(nSamples).encode()
    # because the size of the idid marking dataset is small, data is written to lmdb at one time.
    writeCache(env, cache)
    env.close()

    print(f"LMDB dataset created at {output_path} with {nSamples} samples")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', help='Root dir for image files.')
    parser.add_argument('labels_dir', help='Root dir for label txt files.')
    parser.add_argument(
        'out_dir', help='Dir to save lmdb data.')
    parser.add_argument(
        'train_val_ratio', help='train and val ratio. 0<train_val_ratio<1')

    args = parser.parse_args()
    return args


def process(img_dir, ann_dir, output_lmdb_path, train_val_ratio):
    image_label_list = [
        # ('path/to/image1.png', 'label1'),
        # ('path/to/image2.png', 'label2'),
        # Add more (image_path, label) pairs here
    ]

    img_files = os.listdir(img_dir)
    ann_files = os.listdir(ann_dir)
    assert len(img_files) == len(ann_files)
    total_file_no = len(img_files)
    for i in range(total_file_no):
        full_img_path = os.path.join(img_dir, f'img_{i}.jpg')
        full_annotation_path = os.path.join(ann_dir, f'gt_img_{i}.txt')
        if os.path.exists(full_img_path) and os.path.exists(full_annotation_path):
            with open(full_annotation_path, 'r') as file:
                content = file.read()
                # label not include region points
                label = content.split(",")[-1]
            one_tuple = (full_img_path, label)
            image_label_list.append(one_tuple)

    # shuffle original order of list
    random.shuffle(image_label_list)
    # create train and val mdb files
    create_lmdb_dataset(output_lmdb_path, image_label_list, train_val_ratio)


def read_lmdb_dataset(lmdb_path, num_samples=None, remove_whitespace=True, charset=None, normalize_unicode=True):
    """
    Read data from an LMDB dataset.

    Args:
        lmdb_path (str): Path to the LMDB database.
        num_samples (int, optional): Number of samples to read. If None, read all samples.
        remove_whitespace(bool): whether removing white space from the labels.
        charset: target charset
        normalize_unicode: Normalize unicode composites (if any) and convert to compatible ASCII characters
    """
    charset_adapter = CharsetAdapter(charset)
    # Open the LMDB environment
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        # Get the total number of entries in the LMDB
        num_samples = int(txn.get('num-samples'.encode()))
        print(f"Total samples in LMDB: {num_samples}")

        # Set the number of samples to read
        if num_samples is None or num_samples > num_samples:
            num_samples = num_samples

        for index in range(num_samples):
            index += 1  # lmdb starts with 1

            label_key = f'label-{index:09d}'.encode()
            label = txn.get(label_key).decode()
            # Normally, whitespace is removed from the labels.
            if remove_whitespace:
                print('-----------------------')
                print(f'label:{label}')
                label = ''.join(label.split())
                print(f'label:{label} after removing whitespace')
            if normalize_unicode:
                label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()
            label = charset_adapter(label)

            img_key = f'image-{index:09d}'.encode()
            imgbuf = txn.get(img_key)
            buf = io.BytesIO(imgbuf)
            img = Image.open(buf)
            print(f'image mode:{img.mode}')
            if img.mode != 'RGB':
                img = img.convert('RGB')

            print(img_key)
            print(label_key)
            print(f"Label: {label}")


    # Close the LMDB environment
    env.close()


def main():
    args = parse_args()
    process(args.images_dir, args.labels_dir, args.out_dir, args.train_val_ratio)
    print('finish')

    # read data from lmdb dataset for verification
    # lmdb_path = 'C:\\used_datasets\\idid-marked-icdar15-cropped-1670-lmdb\\val'
    # 94 full
    # charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    # read_lmdb_dataset(lmdb_path, num_samples=10, remove_whitespace=True, charset=charset, normalize_unicode=True)


if __name__ == '__main__':
    main()
