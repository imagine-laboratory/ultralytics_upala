# YOLO Dataset Structure

This README outlines the dataset structure and preparation steps for training a YOLO model.

## Dataset Overview

The dataset is designed for object detection tasks using the YOLO (You Only Look Once) framework. It consists of images and their corresponding label files. The images depict various objects, and the label files contain the bounding box coordinates and class information for each object.

### Folder Structure

The dataset follows the structure below:

```plaintext
dataset/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
└── data.yaml
```

**images/**: Contains all the images used for training, validation, and testing.

- `train/`: Images used for training the model.
- `val/`: Images used for validation during training.
- `test/`: Images used for testing the trained model (optional).

**labels/**: Contains the corresponding label files (in .txt format) for the images.

- `train/`: Label files for training images.
- `val/`: Label files for validation images.
- `test/`: Label files for testing images (optional).~

**data.yaml**: This file contains the configuration details about the dataset, such as class names, number of classes, and the paths to the images and labels.

### Image Format
- **Format**: The images should be in a format that is compatible with the YOLO model (e.g., .jpg, .png).
- **Size**: It is recommended to keep the image sizes consistent or rescale them during training. YOLO typically resizes images to a standard size like 416x416 or 640x640 pixels.

### Label Format
Each image must have a corresponding label file with the same name, located in the `labels/` folder. Label files must be in plain text format (`.txt`), and each line should represent one object in the image with the following structure:
```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: Integer representing the class of the object (starts from 0).
- `x_center`: The x-coordinate of the bounding box center, normalized to [0, 1] relative to the image width.
- `y_center`: The y-coordinate of the bounding box center, normalized to [0, 1] relative to the image height.
- `width`: The width of the bounding box, normalized to [0, 1] relative to the image width.
- `height`: The height of the bounding box, normalized to [0, 1] relative to the image height.

### Example of a label file (image1.txt):

```
0 0.512 0.421 0.145 0.233
1 0.374 0.599 0.180 0.291
```
- In this example, the first object is of class 0 with its bounding box centered at (0.512, 0.421) and size (0.145, 0.233) relative to the image dimensions.

### data.yaml Format
The data.yaml file provides essential information to the YOLO model during training. Below is an example:
```
train: dataset/images/train/
val: dataset/images/val/
test: dataset/images/test/  # Optional

nc: 2  # Number of classes
names: ['background', 'building']  # List of class names

```

- `train`: Path to the folder containing training images.
- `val`: Path to the folder containing validation images.
- `test`: (Optional) Path to the folder containing test images.
- `nc`: Number of object classes in the dataset.
- `names`: A list of class names corresponding to the class_id in the label files.

### Preparing the Dataset
- **Organize Images and Labels**: Place images in the images/train, images/val, and optionally images/test directories. Ensure each image has a corresponding .txt file in the respective labels folder.
- **Create data.yaml**: Ensure the file paths and class information are correctly specified in data.yaml.
- **Check Label Format**: Make sure all label files follow the correct format and normalization.