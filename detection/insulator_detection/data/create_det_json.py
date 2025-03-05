import json


def create_coco_json(merged_ann_json: str, coco_val_json: str):
    # Load data from the JSON file
    with open(merged_ann_json, "r") as infile:
        data = json.load(infile)
    # COCO format structure
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "No issues"},
            {"id": 1, "name": "Broken"},
            {"id": 2, "name": "Flashover damage"},
            # {"id": 4, "name": "notbroken-notflashed"}
        ],
    }

    annotation_id = 1
    for img_obj in data["data"]:
        # image data info format
        image_info = {
            "id": img_obj["id"],
            "file_name": img_obj["file_name"],
            "width": img_obj["width"],
            "height": img_obj["height"],
        }
        coco_format["images"].append(image_info)
        # annotations
        for ann_obj in img_obj["obj_annotations"]:
            # annotation data info format
            annotation = {
                "id": annotation_id,
                "image_id": ann_obj["image_id"],
                "category_id": ann_obj["category_id"],  # if category id is 1, 2, 3, needs to subtract 1
                "bbox": ann_obj["bbox"],
                "area": ann_obj["area"],
                "iscrowd": ann_obj["iscrowd"]
            }
            coco_format["annotations"].append(annotation)
            annotation_id = annotation_id + 1

    # Save the COCO format data to a new JSON file
    with open(coco_val_json, "w") as outfile:
        json.dump(coco_format, outfile, indent=4)

    print(f"Conversion to COCO format completed and saved to {coco_val_json}.")


def main():
    merged_json = "D:\\project\\merge_ds\\val\\annotations\\annotations.json"
    coco_val_json = "D:\project\\coco_val.json"
    create_coco_json(merged_json, coco_val_json)


if __name__ == '__main__':
    main()
