## system env:
1. Ubuntu: 22.04 
2. Python: 3.10.13
3. PyTorch: 2.2.2+cu118
4. C++ Version: 201703

## packages installation
1. pip install fqdn==1.5.1
2. pip install ftfy==6.2.3
3. pip insall hydra-core==1.3.2
4. pip install imageio==2.35.1
5. pip install imgaug==0.4.0
6. pip install lightning-utilities==0.11.7
7. pip install lmdb==1.5.1
8. mmdet: locate detection folder and pip install -v -e.
9. pip install nltk==3.9.1
10. pip install numpy==1.26.4
11. pip install pytorch-lightning==1.9.4
12. pip install scipy
13. pip install shapely==2.0.6
14. pip install tensorboard==2.17.0
15. [DCNv4 installation](../../classification/README.md)

### architecture
![architecture](../insulator_detection/result/archi.jpg)

## Train
bash insulator_det_str.sh

## TEST
1. bash test.sh 0 last.ckpt
2. [test log](../insulator_detection/result/test.txt)


## Inference
bash read.sh 0 checkpoint.ckpt /YOUR TEST IMAGES FOLDER/

## - dataset for defective insulator dataset with marked text 
###train
####annotations
 annotations.json
####cropped_imgs
####images
###val
####annotations
annotations.json
####cropped_imgs
####images
###test

## - annotation structure
```annotation structure
{
    "data":[
            {
                "id": 0,
                "file_name": "130032.jpg",
                "width": 4288,
                "height": 2848,
                "obj_annotations": [
                    {
                        "id": 0,
                        "image_id": 0,
                        "category_id": 0,
                        "bbox": [
                            2128,
                            1194,
                            405,
                            213
                        ],
                        "area": 86265,
                        "iscrowd": 0
                    },
                    ...
                ],
                "label_annotations": [
                       {
                            "id": 0,
                            "image_id": 0,
                            "label": "LOCKE",
                            "vertex_points": [
                                [
                                    2322,
                                    1296
                                ],
                                [
                                    2399,
                                    1292
                                ],
                                [
                                    2402,
                                    1317
                                ],
                                [
                                    2324,
                                    1320
                                ]
                            ],
                             "cropped_img": "130032_0.jpg"
                       },
                       ...
                ]
            }
            ...         
    ],
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
```
