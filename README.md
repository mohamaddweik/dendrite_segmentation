# SEM Dendrite Segmentation

This project implements two different approaches for the semantic segmentation of dendrites in Scanning Electron Microscope (SEM) images: a Classic Computer Vision pipeline and a Deep Learning approach using YOLO.

## Setup
Install the required dependencies using pip:
`pip install -r requirements.txt`

## 1. Classic Computer Vision Approach
To run the deterministic pipeline (Pre-processing -> Sauvola -> Morphology -> Skeletonization):
`python SEMDendriteSegmenter.py` 

## 2. Deep Learning Approach (YOLO)

### Dataset Preparation
To improve segmentation of thin structures, the training images are split into overlapping tiles. Run this to generate the tiled dataset:
`python make_tiled_dataset.py`

### Training
To train the YOLO segmentation model (`yolo26n-seg.pt`):
`python train_yolo_dendrite_seg.py`
Results and the best model weights (`best.pt`) will be saved automatically in the `runs/segment/` directory.

### Inference on Large Images (Tiled Prediction)
For high-resolution images, we use a tiled inference approach with a voting mechanism to merge the masks. 
**Important:** Before running, open `predict_tile.py` and update the following variables with your local paths:
* `weights`: Path to your trained `.pt` model.
* `source_folder`: Path to the folder containing the images you want to predict on.
* `out_dir`: Path where you want the masks and overlays saved.

Once updated, run:
`python predict_tile.py`

### Inference on Standard Images
If your images are smaller and don't require tiling, use the standard prediction script. Like the tiled script, make sure to update your model and image paths inside the file first:
`python predict.py`

### Command Line Inference (Single Image)
If you prefer using the Ultralytics CLI directly on a single image:
`yolo segment predict model="runs/segment/yolo26_dendrite_tiled_v1/weights/best.pt" source="path/to/new_image.png" imgsz=896 conf=0.15 iou=0.80 device=cpu save=True`
