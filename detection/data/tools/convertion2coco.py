import json
from PIL import Image

root = "D:\\Desktop\图像检索\\IDIDataset\idid-coco\\val\\"
# Load data from the JSON file
with open("../jsons/labels_v1.2_val.json", "r") as infile:
    data = json.load(infile)

# COCO format structure
coco_format = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "No issues"},
        {"id": 2, "name": "Broken"},
        {"id": 3, "name": "Flashover damage"},
        # {"id": 4, "name": "notbroken-notflashed"}
    ],
}

# Mapping conditions to category IDs
category_mapping = {
    "No issues": 1,
    "Broken": 2,
    "Flashover damage": 3,
    "notbroken-notflashed": 4
}

annotation_id = 1
image_id = 1

for entry in data:
    # 打开图片文件
    with Image.open(root + entry["filename"]) as img:
        width, height = img.size
        print(f"Width: {width}, Height: {height}")

    image_info = {
        "id": image_id,
        "file_name": entry["filename"],
        "width": width,
        "height": height,
    }
    coco_format["images"].append(image_info)

    for obj in entry["Labels"]["objects"]:
        try:
            x, y, w, h = obj["bbox"]
            # Extracting the condition category
            condition = list(obj["conditions"].values())[0]
            category_id = category_mapping.get(condition, None)
            if category_id:
                if category_id == 4:
                    category_id = 1
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                }
                coco_format["annotations"].append(annotation)
                annotation_id += 1
        except KeyError as e:
            print(f"KeyError: {e} not found in dictionary")
            print('this is an insulator string.')
            print(obj["string"])

    image_id += 1

# Save the COCO format data to a new JSON file
with open("../jsons/labels_v1.2_val_coco.json", "w") as outfile:
    json.dump(coco_format, outfile, indent=4)

print("Conversion to COCO format completed and saved to labels_v1.2_coco.json.")
