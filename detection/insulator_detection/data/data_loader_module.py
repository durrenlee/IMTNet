# modified from Scene Text Recognition Model Hub Copyright 2022 Darwin Bautista
# changed to custom insulator str and detection dataset

import os.path

import torch
from torchvision import transforms as T

import pytorch_lightning as pl
from typing import Optional, Callable, Sequence, Tuple
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import DataLoader

from .dataset.dataset import InsulatorDataset

#  Implement a subclass of pl.LightningDataModule
class InsulatorDataModule(pl.LightningDataModule):
    def __init__(self, root_dir: str, str_img_size: Sequence[int], det_img_size: Sequence[int],
                 max_label_length: int, charset_train: str, charset_test: str,
                 batch_size: int, val_batch_size: int, test_batch_size: int, num_workers: int,
                 rand_augment: bool, remove_whitespace: bool = True, normalize_unicode: bool = True, min_image_dim: int = 0,
                 rotation: int = 0, collate_fn: Optional[Callable] = None, openai_meanstd: bool = True,
                 use_cropped_images: bool = True
                 ):
        super().__init__()
        self.root_dir = root_dir  # dataset root dir
        rank_zero_info("dataset root dir:{}".format(self.root_dir))
        self.train_dir = os.path.join(root_dir, 'train')
        self.val_dir = os.path.join(root_dir, 'val')
        self.test_dir = os.path.join(root_dir, 'test')

        self.str_img_size = tuple(str_img_size)  # input image size for str model
        self.det_img_size = tuple(det_img_size)  # input image size for detection model

        self.max_label_length = max_label_length  # insulator marked text label max length
        self.charset_train = charset_train
        self.charset_test = charset_test

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        self.num_workers = num_workers

        self.rand_augment = rand_augment  # apply rand augment and others
        print(f'self.rand_augment:{self.rand_augment}')
        self.remove_whitespace = remove_whitespace  # remove whitespace between word or character
        self.normalize_unicode = normalize_unicode
        self.min_image_dim = min_image_dim
        self.rotation = rotation  # rotate input image for augment

        # self.collate_fn = collate_fn

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        self.mean = (0.48145466, 0.4578275, 0.40821073) if openai_meanstd else 0.5
        self.std = (0.26862954, 0.26130258, 0.27577711) if openai_meanstd else 0.5

        self.use_cropped_images = use_cropped_images

    # @staticmethod
    # def get_transform(img_size: Tuple[int], rand_augment: bool = False, mean=0.5, std=0.5):
    #     transforms = []
    #     print(f'rand_augment:{rand_augment}')
    #     if rand_augment:
    #         from .augment import rand_augment_transform
    #         transforms.append(rand_augment_transform())
    #
    #     transforms.extend([
    #         T.Resize(img_size, T.InterpolationMode.BICUBIC),
    #         T.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
    #     ])
    #     # ColorJitter, p = 0.2
    #     if torch.rand(1) < 0.2:
    #         transforms.extend([
    #             T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    #         ])
    #     # GaussianBlur, p = 0.2
    #     if torch.rand(1) < 0.2:
    #         transforms.extend([
    #             T.GaussianBlur(kernel_size=3),
    #         ])
    #
    #     # to tensor and normalize
    #     transforms.extend([
    #         T.ToTensor(),
    #         T.Normalize(mean, std)
    #     ])
    #     return T.Compose(transforms)

    @property
    def train_dataset(self):
        # if self._train_dataset is None:
        #     str_transform = self.get_transform(self.str_img_size, self.rand_augment, mean=self.mean, std=self.std)
        #     det_transform = self.get_transform(self.det_img_size, self.rand_augment, mean=self.mean, std=self.std)
        #     self._train_dataset = InsulatorDataset(root_dir=self.train_dir, flip=True, str_transform=str_transform,
        #                                            det_transform=det_transform,
        #                                            use_cropped_images=self.use_cropped_images,
        #                                            det_img_size=self.det_img_size)
        #     rank_zero_info('\tinsulator dataset: The number of training samples is {}'.format(len(self._train_dataset)))
        return self._train_dataset

    @property
    def val_dataset(self):
        # if self._val_dataset is None:
        #     str_transform = self.get_transform(self.str_img_size, self.rand_augment, mean=self.mean, std=self.std)
        #     det_transform = self.get_transform(self.det_img_size, self.rand_augment, mean=self.mean, std=self.std)
        #     self._val_dataset = InsulatorDataset(root_dir=self.val_dir, str_transform=str_transform,
        #                                          det_transform=det_transform,
        #                                          use_cropped_images=self.use_cropped_images,
        #                                          det_img_size=self.det_img_size)
        #     rank_zero_info('\tinsulator dataset: The number of validation samples is {}'.format(len(self._val_dataset)))
        return self._val_dataset

    @property
    def test_dataset(self):
        # if self._test_dataset is None:
        #     str_transform = self.get_transform(self.str_img_size, self.rand_augment, mean=self.mean, std=self.std)
        #     det_transform = self.get_transform(self.det_img_size, self.rand_augment, mean=self.mean, std=self.std)
        #     self._test_dataset = InsulatorDataset(root_dir=self.test_dir, str_transform=str_transform,
        #                                           det_transform=det_transform,
        #                                           use_cropped_images=self.use_cropped_images,
        #                                           det_img_size=self.det_img_size)
        #     rank_zero_info(
        #         '\tinsulator dataset: The number of test samples is {}'.format(len(self._test_dataset)))
        return self._test_dataset

    def train_dataloader(self):
        # self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
        #                   num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
        #                   pin_memory=True, collate_fn=_collate_fn)
        return self.train_data_loader

    def val_dataloader(self):
        # self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.val_batch_size,
        #                   num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
        #                   pin_memory=True, collate_fn=_collate_fn)
        return self.val_data_loader

    def test_dataloader(self):
        # self.test_data_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size,
        #                   num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
        #                   pin_memory=True, collate_fn=_collate_fn)
        return self.test_data_loader

# def _collate_fn(batch):
#     # Separate each field into its own list
#     images = [item['image'] for item in batch]
#     bboxes = [item['bboxes'] for item in batch]
#     labels = [item['labels'] for item in batch]
#     cropped_images = [item['cropped_images'] for item in batch]
#     text_labels = [item['text_labels'] for item in batch]
#     img_metas = [item['img_meta'] for item in batch]
#
#     # Stack images (assuming they are all the same shape)
#     images = torch.stack(images, dim=0)
#
#     # For `bboxes` and `labels`, since the number of bounding boxes may vary per image,
#     # we keep them as lists. This structure is necessary to handle variable-length data.
#
#     # Convert cropped images if available (note they are already transformed)
#     # Stack cropped images for each sample individually, then pad them to maintain batch consistency
#     max_cropped_len = max(len(imgs) for imgs in cropped_images) if cropped_images else 0
#     padded_cropped_images = []
#     for imgs in cropped_images:
#         # Pad each list of cropped images to max length
#         padded_imgs = imgs + [torch.zeros_like(imgs[0])] * (max_cropped_len - len(imgs))
#         padded_cropped_images.append(torch.stack(padded_imgs, dim=0))
#     # Stack the whole batch
#     cropped_images = torch.stack(padded_cropped_images, dim=0) if padded_cropped_images else None
#
#     # Text labels can vary in length, so we keep them as lists of lists
#     text_labels = text_labels
#
#     # Return batched dictionary
#     return {
#         'images': images,
#         'bboxes': bboxes,
#         'labels': labels,
#         'cropped_images': cropped_images,
#         'text_labels': text_labels,
#         'img_metas': img_metas
#     }