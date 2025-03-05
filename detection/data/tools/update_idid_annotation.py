import json

# Load data from the annotations-added json file
with open("../jsons/annotations_added-val2.json", "r") as infile:
    data = json.load(infile)

# COCO format structure
coco_format = {
    "images": [],
    "annotations": []
}

annotation_id = 3632
image_id = 325

for img in data["images"]:
    # for obj in entry["images"]:
    print("image id:")
    print(img["id"])
    image_info = {
        "id": image_id,
        "file_name": img["file_name"][11:],
        "width": img["width"],
        "height": img["height"],
    }
    print(image_info)
    coco_format["images"].append(image_info)

    for ann in data["annotations"]:
        if ann["image_id"] == img["id"]:
            print("    ann id:")
            print("    " + str(ann["id"]))

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": ann["iscrowd"]
            }
            print(annotation)
            coco_format["annotations"].append(annotation)
            annotation_id += 1
    image_id += 1



# Save the COCO format data to a new JSON file
with open("../jsons/new_added-val2.json", "w") as outfile:
    json.dump(coco_format, outfile, indent=4)