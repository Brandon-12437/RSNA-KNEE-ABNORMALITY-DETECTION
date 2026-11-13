# RSNA KNEE ABNORMALITY DETECTION

## Problem Description

Knee abnormalities are common musculoskeletal conditions that can affect mobility, physical function, and quality of life.

Magnetic Resonance Imaging (MRI) is widely used to examine knee structures and identify abnormalities involving ligaments, menisci, cartilage, and surrounding tissues.

However, interpreting MRI scans requires medical expertise and can be time-consuming, particularly when examining multiple MRI series across different anatomical planes.
The Solution
The RSNA Knee Abnormality Detection system uses deep learning and machine learning techniques to identify potential knee abnormalities from MRI images.

The system processes MRI DICOM images and uses an EfficientNetB0 convolutional neural network for feature extraction.

The extracted image features are then passed to a Logistic Regression classifier to generate predictions for 12 knee abnormality categories.

The trained prediction pipeline is exposed through a Flask REST API, allowing users to submit MRI studies packaged as ZIP files and receive prediction results.
