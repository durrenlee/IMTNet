import json
from PIL import Image

# Load data from the JSON file
with open("../jsons/labels_v1.2_val_coco.json", "r") as infile:
    data = json.load(infile)

for img_obj in data["images"]:
    # print(img_obj['id'])

    has_annotation = 0
    for annotation in data["annotations"]:
        if annotation["image_id"] == img_obj['id']:
            # print(img_obj['id'])
            has_annotation = 1
            break

    if has_annotation == 1:
        # print(img_obj['id'])
        pass
    else:
        print(print(img_obj['id']))