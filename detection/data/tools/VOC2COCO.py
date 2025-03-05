import os
import json
import xml.etree.ElementTree as ET
from PIL import Image


def get_categories(voc_dir):
    categories = []
    annotations_dir = os.path.join(voc_dir, 'Annotations')
    xml_files = [os.path.join(annotations_dir, file) for file in os.listdir(annotations_dir) if file.endswith('.xml')]
    unique_categories = set()

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('object'):
            category = obj.find('name').text
            unique_categories.add(category)

    for idx, category in enumerate(sorted(unique_categories)):
        categories.append({
            "id": idx + 1,
            "name": category,
            "supercategory": "none"
        })
    return categories


def convert_voc_to_coco(voc_dir, output_file):
    annotations_dir = os.path.join(voc_dir, 'Annotations')
    images_dir = os.path.join(voc_dir, 'JPEGImages')
    xml_files = [os.path.join(annotations_dir, file) for file in os.listdir(annotations_dir) if file.endswith('.xml')]

    images = []
    annotations = []
    categories = get_categories(voc_dir)
    category_map = {category['name']: category['id'] for category in categories}

    annotation_id = 1
    for img_id, xml_file in enumerate(xml_files, 1):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        img_path = os.path.join(images_dir, filename)

        # Check image size
        img = Image.open(img_path)
        width, height = img.size

        # Add image to COCO dataset
        images.append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        for obj in root.findall('object'):
            category = obj.find('name').text
            category_id = category_map[category]

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            o_width = xmax - xmin
            o_height = ymax - ymin

            annotations.append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "area": o_width * o_height,
                "iscrowd": 0
            })
            annotation_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Save COCO formatted dataset
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=4)

    print(f"COCO dataset saved to {output_file}")


# Usage example
voc_dir = "/path/to/VOC"  # Path to VOC dataset
output_file = "/path/to/output/coco_annotations.json"
convert_voc_to_coco(voc_dir, output_file)
