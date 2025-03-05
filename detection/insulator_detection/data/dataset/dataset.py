import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os

from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.utils import _log_api_usage_once
import random

class InsulatorDataset(Dataset):
    def __init__(self, root_dir, flip=False, str_transform=None, det_transform=None, use_cropped_images=True, det_img_size=None):
        """
        Args:
            root_dir (str): Root directory of the dataset (train or val folder).
            transform (callable, optional): Optional transform to be applied on an image.
            use_cropped_images (bool): Whether to use pre-cropped images for text recognition.
        """
        self.root_dir = root_dir
        self.flip = flip
        self.str_transform = str_transform
        self.det_transform = det_transform
        self.use_cropped_images = use_cropped_images
        self.det_img_size = det_img_size
        # Load the annotations.json file
        annotations_path = os.path.join(root_dir, 'annotations', 'annotations.json')
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)['data']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get image information
        image_info = self.annotations[idx]
        image_id = image_info['id']
        file_name = image_info['file_name']

        # Load the original image
        img_path = os.path.join(self.root_dir, 'images', file_name)
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = Image.open(img_path).convert("RGB")

        # Get defect annotations (bounding boxes)
        obj_annotations = image_info.get('obj_annotations', [])
        bboxes = [ann['bbox'] for ann in obj_annotations]
        labels = [ann['category_id'] for ann in obj_annotations]

        # Get text recognition annotations
        label_annotations = image_info.get('label_annotations', [])
        cropped_images = []
        text_labels = []
        if self.use_cropped_images:
            # Load pre-cropped images for text recognition
            for label_ann in label_annotations:
                cropped_img_path = os.path.join(self.root_dir, 'cropped_imgs', label_ann['cropped_img'])
                cropped_image = Image.open(cropped_img_path)
                if cropped_image.mode != "RGB":
                    cropped_image = cropped_image.convert("RGB")
                cropped_images.append(cropped_image)
                text_labels.append(label_ann['label'])

        # Convert to tensor
        bboxes = torch.tensor(bboxes)
        labels = torch.tensor(labels)

        # apply h or v flip based on probability given with 0.5
        flip = False
        flip_direction = None
        if self.flip:
            if torch.rand(1) < 0.5:
                hv_list = ['Horizontal', 'Vertical']
                random_element = random.choice(hv_list)
                if random_element == 'Horizontal':
                    image = F.hflip(image)
                    flip_direction = 'Horizontal'
                else:
                    image = F.vflip(image)
                    flip_direction = 'Vertical'
                flip = True

        # Apply transformations if specified
        if self.det_transform is not None:
            image = self.det_transform(image)
        if self.str_transform is not None:
            cropped_images = [self.str_transform(cropped_img) for cropped_img in cropped_images]

        full_file_name = os.path.join(self.root_dir, 'images')
        full_file_name = os.path.join(full_file_name, file_name)
        img_meta = {
            'img_id': image_id,
            'img_path': full_file_name,
            'ori_shape': (image_info['height'], image_info['width']),   # (height, width)
            'img_shape': (self.det_img_size[0], self.det_img_size[1]),  # (h, w)
            'scale_factor': (self.det_img_size[0] / image_info['height'], self.det_img_size[1] / image_info['width']),
            'flip': flip,
            'flip_direction': flip_direction,
            'pad_shape': (self.det_img_size[0], self.det_img_size[1])
        }
        return {
            'image': image,
            'bboxes': bboxes,
            'labels': labels,
            'cropped_images': cropped_images,
            'text_labels': text_labels,
            'img_meta': img_meta
        }


# for test
if __name__ == "__main__":
    root = "D:\\project\\merge_ds\\train"
    dataset = InsulatorDataset(root, str_transform=None, det_transform=None, use_cropped_images=True, det_img_size=[1333, 800])

    id1, id2 = 90, 136
    image1 = dataset[id1]
    image2 = dataset[id2]
    for label in image1["text_labels"]:
        print(label)
    print(f'img_meta:{image1["img_meta"]}')
    print('------------------------')
    for label in image2["text_labels"]:
        print(label)

    # output_path = "D:\project\output"
    # image1["image"].save(os.path.join(output_path, "imiage_{}.jpg".format(id1)))
    # image2["image"].save(os.path.join(output_path, "imiage_{}.jpg".format(id2)))

    # idx = 0
    # for cropped_img in image1["cropped_images"]:
    #     cropped_img.save(os.path.join(output_path, f'{image1["image_id"]}-{idx}.jpg'))
    #     idx = idx + 1
    #
    # idx = 0
    # for cropped_img in image2["cropped_images"]:
    #     cropped_img.save(os.path.join(output_path, f'{image2["image_id"]}-{idx}.jpg'))
    #     idx = idx + 1


