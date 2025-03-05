###### idid det-str dataset creation steps:
1. use labelme to label defect insulators with rectangle bbox.
2. use labelme2coco.py to convert labelme format to coco-based json, image files and visualized images.
3. use train_val_split.py to split coco-based json and images into train and val dataset.
4. use labelme to label insulator markings with four vertexes bbox.
5. use label2icdar15.py to convert labelme format to icdar 2013/2015 dataset. 
   1. task: 'crop' is used to create cropped images and label text files. Each cropped image has own label file.
   2. task: 'standard' is used to create label files and original image files, one original image only has one label file contains multiple lines with bbox vertexes and text content.
6. use combine_coco_icdar15_idid.py to combine insulator idid str id15-based dataset and defect coco-based dataset together.
7. folder structure of dataset:
   1. train
      1. images # original images
      2. cropped_imgs # cropped images
      3. annotations # annotations.json
         1. annotations.json
   2. val
      1. images # original images
      2. cropped_imgs # cropped images
      3. annotations # annotations.json