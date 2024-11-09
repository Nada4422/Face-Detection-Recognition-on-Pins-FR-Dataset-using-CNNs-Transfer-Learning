# Face Detection & Recognition on Pins FR Dataset using CNNs & Transfer Learning
## 1. Objective
The objective of this project is to develop a **real-time face recognition system** capable of detecting and recognizing faces across 105 classes (celebrities) using **Convolutional Neural Networks (CNNs)** and **Transfer Learning** techniques. The system leverages OpenCV’s DNN-based face detector to crop faces, followed by a CNN classifier that identifies the celebrity class with a validation accuracy of 30%.

## 2. Dataset Description
The dataset used for this project is the **Pins FR Dataset** comprising images of 105 celebrities, with each celebrity’s photos organized into individual subfolders within a primary folder called 105_classes_pins_dataset. The dataset will be split as follows:

**1- Training set:** 70% of images per celebrity

**2- Validation set:** 15% of images per celebrity

**3- Testing set:** 15% of images per celebrity
## 3. Project Workflow
The project workflow can be divided into several key steps:

**1- Dataset Preprocessing:**

Loading the dataset and organizing images into training, validation, and testing splits.

Resizing images and applying necessary transformations to standardize input sizes for the CNN model.

**2- Face Detection:**

Using OpenCV’s DNN-based face detector to localize and crop faces from images.

Preparing these cropped faces as inputs for the classification model.

**3-Model Architecture:**

Implementing Transfer Learning with pre-trained InceptionV3 model.

Adding additional dense layers and dropouts to improve accuracy.

**4- Training:**

Training the model on the training set, with hyperparameter tuning to optimize performance.

**5- Validation and Testing:**

Evaluating the model on both validation and testing datasets to assess generalization performance.

Recording performance metrics and generating visualizations for model evaluation.
## 4. Getting Started
**1- Dependencies**

Ensure the following dependencies are installed:

Python 3.11.10

OpenCV

TensorFlow

Keras

Numpy

Matplotlib

Jupyter Notebook (Anaconda environment recommended for local runs)

**2- Installation**

Install dependencies by running:

pip install -r requirements.txt

## 5. Running the Code
**1- Upload Dataset:**

Upload the **105_classes_pins_dataset** folder to your environment. 

**2- Execute Notebook:** 

Open and run each cell of the **Face_Detection_Recognition.ipynb** notebook. The code is organized into sections for:

Preprocessing and dataset split.

Face detection using OpenCV.

Model training with Transfer Learning and hyperparameter tuning.

Validation and testing evaluation.
## 6. Expected Results and Target Accuracy
The objective is to achieve 30%+ accuracy on the validation set. Detailed performance results, including accuracy and loss curves, confusion matrix, and classification reports, are provided for both the validation and testing datasets.

## 7. Evaluation Metrics and Visualizations
Comprehensive evaluation includes:

**1- Accuracy and Loss Curves:** Visual representation of model performance over epochs.

**2- Confusion Matrix:** Helps analyze per-class accuracy.

**3- Precision, Recall, F1-Score:** Provided in a classification report for deeper analysis.

**4- Model Metrics on Validation and Test Sets:** Displayed as graphs for intuitive understanding.