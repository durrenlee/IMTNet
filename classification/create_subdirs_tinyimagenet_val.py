import os
import shutil

# Paths
val_dir = "/root/data/tiny-imagenet-200/val"
val_img_dir = os.path.join(val_dir, "images")
annotations_file = os.path.join(val_dir, "val_annotations.txt")

# Create class subdirectories in `val`
with open(annotations_file, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        image_filename = parts[0]
        class_label = parts[1]
        class_dir = os.path.join(val_dir, class_label)

        # Create class directory if it doesn't exist
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # Move image to the class directory
        src_path = os.path.join(val_img_dir, image_filename)
        dst_path = os.path.join(class_dir, image_filename)
        shutil.move(src_path, dst_path)

# Remove the original `images` directory (now empty)
os.rmdir(val_img_dir)

print("Validation set reorganized successfully.")
