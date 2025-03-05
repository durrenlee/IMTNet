import os
import shutil

idid_coco_imgs_dir_train = "D:\\Desktop\\图像检索\\IDIDataset\\idid-coco\\train"
idid_coco_imgs_dir_val = "D:\\Desktop\\图像检索\\IDIDataset\\idid-coco\\val"
str_imgs_dir = "D:\project\带型号标记的缺陷绝缘子\labelme\img_data"

file_names_idid_train = os.listdir(idid_coco_imgs_dir_train)
file_names_idid_val = os.listdir(idid_coco_imgs_dir_val)
target_dir = "C:\\used_datasets\\labelme\\img_data"

for str_img_name in os.listdir(str_imgs_dir):
    # print(str_img_name)
    if str_img_name not in file_names_idid_train and str_img_name not in file_names_idid_val:
        print(str_img_name)
        # copy to temp dir
        image_path = os.path.join(str_imgs_dir, str_img_name)
        to_image_path = os.path.join(target_dir, str_img_name)
        shutil.copy2(image_path, to_image_path)