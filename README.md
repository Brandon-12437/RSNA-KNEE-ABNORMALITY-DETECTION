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
<img width="700" alt="EDA Analysis" src="screenshots/Screenshot%20from%202026-11-13%2015-26-51.png" />

## Model Training and Reproducibility

### Data Splitting

The 58 fully labelled MRI studies were divided into training and validation sets using iterative stratification to maintain the distribution of the 12 target labels.

- Training studies: 47
- Validation studies: 11

### Feature Extraction

A pretrained EfficientNetB0 model was used as a feature extractor.

- Pretrained weights: ImageNet
- Input dimensions: 224 × 224 × 3
- Model configuration: include_top=False
- Feature extraction: Global Average Pooling
- Extracted feature dimensions: 1,280

The EfficientNetB0 layers were frozen, allowing the model to extract image features without updating its pretrained weights.

### Classification

Logistic Regression was used to perform multi-label classification using the extracted image features. The classifier generated predictions for the 12 knee abnormality labels.

### Reproducibility

The trained model components were prepared for deployment and integration with the Flask prediction API.

## Containerization with Docker

Docker was used to containerize the RSNA Knee MRI Prediction API, packaging the application code, model dependencies, and runtime environment into a portable container.

### Docker Configuration

The application was configured using:

- **Base image:** Python slim image
- **API framework:** Flask
- **Application server:** Gunicorn
- **Containerization:** Docker
- **Application port:** 5000

### Building the Docker Image

The Docker image was built using the Dockerfile:

                         docker build -t rsna-knee-api .

### Running the Container

The container was started with:

                                 docker run -d --name rsna-knee-container -p 5000:5000 rsna-knee-api

<img width="700" alt="AWS Deployment Screenshot" src="screenshots/Screenshot%20from%202026-11-11%2011-39-23.png" />
