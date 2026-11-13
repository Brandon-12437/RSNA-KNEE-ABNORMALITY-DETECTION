# RSNA KNEE ABNORMALITY DETECTION

## Problem Description

Knee abnormalities are common musculoskeletal conditions that can affect mobility, physical function, and quality of life.

Magnetic Resonance Imaging (MRI) is widely used to examine knee structures and identify abnormalities involving ligaments, menisci, cartilage, and surrounding tissues.

However, interpreting MRI scans requires medical expertise and can be time-consuming, particularly when examining multiple MRI series across different anatomical planes.
## The Solution
The RSNA Knee Abnormality Detection system uses deep learning and machine learning techniques to identify potential knee abnormalities from MRI images.

The system processes MRI DICOM images and uses an EfficientNetB0 convolutional neural network for feature extraction.

The extracted image features are then passed to a Logistic Regression classifier to generate predictions for 12 knee abnormality categories.

The trained prediction pipeline is exposed through a Flask REST API, allowing users to submit MRI studies packaged as ZIP files and receive prediction results.

## DATASET EXPLORATION
The project uses the RSNA Knee MRI dataset provided through the Kaggle competition.

The dataset contains MRI studies stored in DICOM format, alongside CSV files containing study information and abnormality labels.
### Dataset Structure
The dataset includes:
- Training MRI studies.
- Testing MRI studies.
- DICOM image series.
- Study-level metadata.
- Abnormality labels.
- Sample submission file.
The training metadata contains 4,407 studies and 14 columns.

However, only 58 studies contained complete abnormality labels suitable for the supervised learning workflow implemented in this project.
Exploratory Analysis
The dataset exploration involved:
- Inspecting the training and testing CSV files.
- Examining study identifiers.
- Investigating MRI series directories.
- Examining anatomical planes.
- Inspecting image acquisition characteristics.
- Identifying studies with available labels.
- Examining the distribution of the 12 target abnormalities.
- Visualizing selected MRI slices.
Target Abnormalities
The model predicts the following 12 abnormalities:
1. ACL
2. MCL
3. Medial Meniscus
4. Lateral Meniscus
5. Medial Osteoarthritis
6. Lateral Osteoarthritis
7. Patellofemoral Osteoarthritis
8. Effusion
9. Synovitis
10. Baker's Cyst
11. Contusion
12. Fracture
    
# DATA VISUALIZATION
<img width="700" alt="Dataset Overview" src="screenshots/Screenshot%20from%202026-11-13%2012-47-24.png" />
