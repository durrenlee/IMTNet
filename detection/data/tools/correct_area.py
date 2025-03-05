import json

# Load data from the annotations-added json file
with open("../jsons/labels_v1.2_val_coco.json", "r") as infile:
    data = json.load(infile)

# COCO format structure
coco_format = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "No issues"},
        {"id": 2, "name": "Broken"},
        {"id": 3, "name": "Flashover damage"},
    ],
}

for img in data["images"]:
    image_info = {
        "id": img["id"],
        "file_name": img["file_name"],
        "width": img["width"],
        "height": img["height"],
    }
    print(image_info)
    coco_format["images"].append(image_info)

for ann in data["annotations"]:
    x, y, w, h = ann["bbox"]
    annotation = {
        "id": ann["id"],
        "image_id": ann["image_id"],
        "category_id": ann["category_id"],
        "bbox": ann["bbox"],
        "area": w * h,
        "iscrowd": 0
    }
    coco_format["annotations"].append(annotation)


# Save the COCO format data to a new JSON file
with open("../jsons/labels_v1.2_val_coco_final.json", "w") as outfile:
    json.dump(coco_format, outfile, indent=4)

print("job finished!!!")