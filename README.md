Dental Disease Detection – Dataset Preparation and Annotation
Project Overview

This project focuses on the preparation of a structured dataset for automated dental disease detection using deep learning–based object detection models in YOLO format with CNN classification. The primary objective is to preprocess dental images, manually annotate disease-affected regions, and generate properly formatted label files for training object detection models.

The system handles five dental diseases: Calculus, Dental Caries, Hypodontia, Tooth Discoloration, and Gingivitis. Each disease is treated as a separate detection class. Due to the large size of the complete dataset, only sample images and their corresponding label files are uploaded to this repository. The full dataset is maintained locally.


Diseases Covered

This dataset includes five major dental conditions.

Calculus refers to hardened dental plaque (tartar) that accumulates on teeth and may lead to gum disease. The affected areas are manually annotated using bounding boxes to help the model learn tartar accumulation patterns.

Dental Caries, commonly known as tooth decay, appears as dark or damaged areas on teeth surfaces. Regions containing visible decay were carefully marked using bounding boxes to assist the detection model in identifying decay features.

Hypodontia is a condition involving the congenital absence of one or more teeth. In such cases, bounding boxes are placed around the region where the tooth is missing to allow the model to detect missing-tooth areas.

Tooth Discoloration includes abnormal color changes such as yellowing, browning, or black staining. Affected tooth regions were manually annotated to enable detection of discoloration patterns.

Gingivitis is inflammation of the gums, typically visible as redness or swelling near the gum line. The inflamed gum areas were annotated to help the model distinguish between healthy and inflamed gum tissue.



Image Preprocessing

To ensure consistency and improve model performance, a preprocessing pipeline was applied to all images before annotation.

All images were resized to 240 × 240 pixels. This ensures uniform input dimensions for training, reduces computational complexity, and maintains compatibility with YOLO-based models.

Noise reduction was performed using bilateral filtering. This method reduces image noise while preserving important edges such as tooth boundaries and gum lines. Preserving edges is critical for accurate annotation and feature extraction.

Contrast enhancement was carried out using Contrast Limited Adaptive Histogram Equalization (CLAHE) in the LAB color space. This improves local contrast, enhances visibility of dental structures, and makes disease regions more distinguishable.

Image sharpening was applied using a convolution-based sharpening kernel. This enhances edges and structural details of teeth and gums, improving the clarity of disease regions and increasing annotation precision.

Preprocessed images are saved in separate folders for each disease along with corresponding preprocessed labels.



Annotation Methodology

A custom manual annotation tool was developed using OpenCV. The annotation process is fully interactive and allows bounding boxes to be drawn using mouse drag operations. The tool supports moving existing bounding boxes, undoing the last drawn box, rotating bounding boxes for better alignment, and navigating between images.

Internally, bounding boxes are represented as center coordinates, width, height, and rotation angle. This allows support for oriented bounding boxes. However, for compatibility with YOLO object detection models, labels are saved in standard YOLO format.

Each label file contains entries in the following structure:

<class_id> <x_center> <y_center> <width> <height>

All coordinate values are normalized between 0 and 1 relative to the resized image dimensions. This ensures direct compatibility with YOLO-based training frameworks.



Dataset Organization

Each disease has its own structured directory containing sample original images, corresponding YOLO label files, preprocessed images, and preprocessed labels. The repository includes the annotation and preprocessing scripts required to reproduce the dataset preparation pipeline.

Because the complete dataset contains a large number of images and label files that exceed GitHub’s storage limits, only representative samples are uploaded. The provided code allows the entire dataset to be regenerated locally using the same preprocessing and annotation workflow.


NOTE !!!

The whole imagery dataset with labels for each disease is saved in the laptop only samples are being uploaded beacuse of too much images and labels + the code is also uploaded in the Preprocessed_Data directory with each disease.
