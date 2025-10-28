# YOLO Auto Annotator

## Overview
YOLO Auto Annotator is a tool to automatically generate YOLO-format annotations for a given set of images. 
It uses object detection models to label images and store the results in the YOLO annotation format, making it 
easy to prepare datasets for training YOLO models.

## Requirements
- Python 3.8+
- Jupyter Notebook
- OpenCV
- PyTorch
- Ultralytics YOLOv5 or YOLOv8

## Folder Structure
```

yolo_auto_annotator/
│
├── yolo_auto_annotator.ipynb       # Main Jupyter Notebook containing the code
│
├── dataset/                        # Folder for input images or datasets
│   ├── images/                     # Preproccessed images
│   ├── labels/                     # Labels for preproccessed images
│
├── new_images/                     # Folder for images ready for proccessing
│   ├── labels/                     # Folder for model-created labels
│   ├── pred_vis/                   # Folder for detection Visual output
│
├── transfer_log/                   # Folder for photo transfer log files
│
├── runs/detect/                    # Folder for trained models
│
├── to_save.txt                     # Text file to specify which proccessed photos to save
│
├── data.yaml                       # Yaml file for the model
│
└── README.md                       # Project documentation

```

## How to Run Notebook

### Step 1
Run all code blocks until step 3 (including).

### Step 2
Put all images for proccesing in `new_images` folder.

### Step 3
Run step 4.

### Step 4 -- old
Manually go through all photos in `new_images/pred_vis` folder.</br>
For every photo with correct annotation, write the file name into `to_save.txt` (seperate names with newlines!!!).

### Step 4 -- new
upload the images to cvat</br>
run the cvat_connect script, upload the zipped cvat_labels folder as ultralytics YOLO detection format.</br>
annotate the images as needed,</br>
then download the annotations in ultralytics YOLO detection format </br>
and place them in `new_images/labels` folder with the same file names as the images in `new_images` folder. </br>
run the code block that creates Blank txt files created for all images without annotations (2nd code block in step 4).

> while annotating, if you find an image that is not suitable for the dataset, delete its name from `to_save.txt` so it won't be moved to the train folder in step 5 and delete after (or put them in a separate folder for later). </br>
> Also, if you find an image you do not know how to annotate, leave it out of `to_save.txt` as well and save in the cant_label folder for reexamination.

### Step 5
Run step 5.</br>
All correct images and corresponding labels will be moved to the train folder, and they will be renamed and logged.</br>
Also, `to_save.txt`, and everything left in `new_images/pred_vis` and `new_images/labels` will be cleared.

Note! Incorrectly labled images will stay in `new_images` to be proccess again next iteration.

### Step 6
Repeat (for step 1 you can skip rerunning parts 1 and 2 in the notebook).

## Notes
- Ensure you have a GPU-enabled environment for faster annotation, (otherwise, change `DEVICE = 0` to `DEVICE = "cpu"` in code block under step 1).
- Update the model path in the notebook if you are using custom weights or want to reuse an old model.

---
