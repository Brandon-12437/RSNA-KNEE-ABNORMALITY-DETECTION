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

### API Development and Testing
The trained knee MRI classification model was integrated into a REST API using Flask to enable external applications to submit MRI studies and receive prediction results.
API Implementation
The API provides two main endpoints:
Method	Endpoint	Description
GET	/	Checks API health and availability.
POST	/predict	Accepts an MRI study ZIP file and returns predictions.


The prediction endpoint:
- Accepts MRI DICOM files packaged in a ZIP archive.
- Processes the uploaded MRI study.
- Extracts image features using the pretrained EfficientNetB0 model.
- Passes extracted features to the trained classification model.
- Returns probabilities for 12 knee abnormalities.
Running the API
The API was served using Gunicorn as the production WSGI server and containerized using Docker.

                  docker run -d -p 5000:5000 --name rsna-knee-api rsna-knee-predictor

API Health Check
The root endpoint was tested using:

                  curl -i http://localhost:5000/

Expected response:

          {
             "message": "RSNA Knee MRI Prediction API", 
             "number_of_labels": 12,
             "status": "running"
           }

Screenshot:
<img width="700" alt="API Development" src="screenshots/Screenshot%20from%202026-11-13%2013-35-41.png" />

Prediction Endpoint Testing
The /predict endpoint accepts an MRI study packaged as a ZIP file.

Example request:

       curl -X POST http://localhost:5000/predict \
          -F "file=@knee_mri_study.zip"

The API processes the uploaded study and returns prediction probabilities for the 12 target abnormalities.


<img width="700" alt="API Testing" src="screenshots/Screenshot%20from%202026-11-13%2011-46-29.png" />

### AWS EC2 API Testing

After deploying the Docker container to AWS EC2, the public API endpoint was tested:

               curl -i http://13.50.105.46:5000/
## CLOUD DEPLOYMENT

The RSNA Knee MRI Disease Detection application is deployed on Amazon Web Services (AWS) using an EC2 instance.
Backend API — AWS EC2
The Flask API and trained knee abnormality classification model are deployed on an Ubuntu EC2 instance.

The Dockerized application is hosted on EC2 at:

                            http://13.50.105.46:5000

The /predict endpoint accepts MRI studies in ZIP format, processes the DICOM images, and returns prediction probabilities for 12 knee abnormalities.
Cloud Infrastructure
- Cloud Provider: Amazon Web Services (AWS)
- Compute Service: Amazon EC2
- Operating System: Ubuntu Server
- Containerization: Docker
- API Server: Gunicorn
- Application Port: 5000
  
 <img width="700" alt="AWS EC2 Cloud Deployment" src="screenshots/Screenshot%20from%202026-11-13%2012-52-29.png" />
 
  
<img width="700" alt="AWS Deployment Screenshot" src="screenshots/Screenshot%20from%202026-11-14%2017-40-46.png" />
