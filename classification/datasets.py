# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    print(f'transform:{transform}')

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train='train' if is_train else 'test', transform=transform,
                                    download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)

        print(f'dataset[0] shapes:{dataset[0][0].shape}, {dataset[0][1]}')
        # nb_classes = 1000
        # tiny imagenet dataset
        nb_classes = 200
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'STL-10':
        dataset = datasets.stl10.STL10(args.data_path, split='train' if is_train else 'test', transform=transform,
                                       download=True)
        nb_classes = 10
    elif args.data_set == 'Fashion-MNIST':
        dataset = datasets.mnist.FashionMNIST(args.data_path, train='train' if is_train else 'test',
                                              transform=transform, download=True)
        nb_classes = len(datasets.mnist.FashionMNIST.classes)
    elif args.data_set == 'EMNIST':
        dataset = datasets.mnist.EMNIST(args.data_path, split='balanced', train=is_train,
                                        transform=transform, download=True)
        nb_classes = 47
    elif args.data_set == 'SVHN':
        dataset = datasets.svhn.SVHN(args.data_path, split='train' if is_train else 'test',
                                     transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'CALTECH101':
        dataset = datasets.Caltech101(args.data_path,
                                      transform=transform, download=True)
        nb_classes = 101
    elif args.data_set == 'FLOWERS102':
        dataset = datasets.flowers102.Flowers102(args.data_path, split='train' if is_train else 'val',
                                                 transform=transform, download=True)
        nb_classes = 102
    elif args.data_set == 'IMGTTE':
        dataset = datasets.Imagenette(args.data_path, split='train' if is_train else 'val',
                                      transform=transform, download=False)
        nb_classes = 10

    return dataset, nb_classes


def build_transform(is_train, args):
    print(f'args.input_size:{args.input_size}')
    print(f'is_train:{is_train}')
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        # special conversion for EMNIST
        # transform.transforms.insert(0, transforms.Grayscale(num_output_channels=3))

        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    # special conversion for EMNIST
    # t.append(transforms.Grayscale(num_output_channels=3))

    if resize_im:
        size = int((256 / 224) * args.input_size)
        print(f'size:{size}')
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    return transforms.Compose(t)
