# Handwritten Digit OCR using Statistical Machine Learning

This repository contains an implementation of an OCR (Optical Character Recognition) system for handwritten digit classification based on statistical machine learning methods, developed as part of a diploma thesis at the School of Electrical Engineering, University of Belgrade.

The project focuses on classical pattern-recognition techniques with handcrafted features and compares:

- Linear Bayesian Classifier (LDA)

- k-Nearest Neighbors (kNN)


## Project Overview

The goal is to recognize handwritten digits (0–9) from scanned images by:

1) Image preprocessing and normalization

2) Handcrafted feature extraction

3) Training statistical classifiers

4) Evaluating and comparing performance


Image Preprocessing

The preprocessing pipeline includes:

- Gaussian blur (noise reduction)

- Grayscale to binary thresholding

- Optional erosion and dilation

- Cropping padding/margins

- Resizing to fixed size (e.g., 32x32)

- Optional skeletonization

This ensures all digit samples are standardized before feature extraction.

Example pipeline function:
processing_pipeline(image, target_size=(32, 32))

There is also a visualization utility to save intermediate preprocessing steps.

Feature Extraction

Implemented in extract_features.py, features are handcrafted to describe digit shape and structure, including:

Horizontal and vertical black pixel densities (segmented)

Mean and variance of densities

Image central moments

Number of horizontal and vertical transitions

Radial distance of center of mass from image center

Final feature vector example:
[mean_x, var_x, var_y, moment, h_trans, v_trans, r_center]

Z-score normalization is applied using statistics from the training set.

Main function:
compute_features(img, kx=4, ky=4)

Classifiers

Linear Bayesian Classifier (LDA)

Implemented in lda.py:

Estimates class priors, means, and shared covariance

Computes discriminant functions

Performs multiclass prediction

Key functions:
lda_fit(X, y)
lda_predict(X, model)

Includes accuracy, confusion matrix, per-class precision, and PCA plus pairwise LDA visualization.

k-Nearest Neighbors (kNN)

Supports Euclidean and statistical (Mahalanobis) distance

Optional distance-weighted voting

Stratified k-fold cross-validation for selecting k

Includes accuracy, confusion matrix, classification report, k-fold error plots, and PCA-based 2D visualization.

Evaluation

Both classifiers are evaluated using:

Overall accuracy

Confusion matrices

Per-class precision

Stratified k-fold cross-validation

PCA visualization for feature space insight

Results and analysis are discussed in detail in the thesis (Chapter 5).

## Requirements

Python 3.8+
NumPy
OpenCV
scikit-image
matplotlib

Install dependencies:
pip install numpy opencv-python scikit-image matplotlib

Typical Workflow

Preprocess images using image_processing.py

Extract features using extract_features.py

Normalize features (z-score)

Train and test classifiers:
lda.py
knn.py

Analyze accuracy and confusion matrices

Academic Context

This project was developed as a diploma thesis:

University of Belgrade – School of Electrical Engineering
Department of Signals and Systems
Mentor: Prof. Dr Željko Đurović
Candidate: Uroš Vojicić
October 2025

License

This project is intended for academic and educational purposes.
Feel free to use and modify with proper attribution.

Contact

Uroš Vojicić
Belgrade, Serbia
MSc student in Machine Learning / Computer Engineering
