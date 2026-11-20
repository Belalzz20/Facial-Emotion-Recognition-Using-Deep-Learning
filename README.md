
# Facial Emotion Recognition Using Deep Learning

### Computer Vision | CNN | TensorFlow | Keras | OpenCV

## Overview

This project implements a facial expression classification system using Deep Learning and Computer Vision.

A custom Convolutional Neural Network (CNN) is trained on the FER2013 dataset to classify facial expressions into seven categories.

The project covers the complete machine learning workflow, including dataset preparation, image preprocessing, data augmentation, CNN training, model evaluation, and predictions on custom images.

## Key Features

- Facial expression classification using a CNN
- Seven facial expression categories
- Face detection using OpenCV Haar Cascades
- Grayscale image preprocessing
- CLAHE contrast enhancement
- Gaussian filtering and image sharpening
- Image resizing and normalization
- Data augmentation during model training
- CNN architecture with Batch Normalization and Dropout
- Training and validation performance visualization
- Confusion matrix and classification report
- Custom image predictions with confidence scores

## Facial Expression Categories

The model classifies facial expressions into seven categories:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

## Dataset

The project uses the **FER2013 (Facial Expression Recognition 2013)** dataset.

The dataset contains facial images organized into training and testing directories.

**Dataset Source:**

https://www.kaggle.com/datasets/msambare/fer2013

The dataset is downloaded using KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download("msambare/fer2013")
```

### Dataset Preparation

The preparation workflow includes:

- Loading training and testing images
- Grayscale conversion
- Face detection and cropping
- Image enhancement
- Resizing images to 128 × 128 pixels
- Preparing labels for multi-class classification
- Splitting data for training and validation

## Image Preprocessing

The project applies several image processing techniques before passing images to the CNN.

### 1. Grayscale Conversion

Input images are converted to grayscale to reduce image complexity.

### 2. Face Detection

OpenCV Haar Cascade classifiers are used to locate facial regions.

When a face is detected, the first detected face is cropped for further processing.

### 3. Image Resizing

Images are resized to a consistent resolution of 128 × 128 pixels.

### 4. CLAHE

Contrast Limited Adaptive Histogram Equalization is applied to improve local contrast.

### 5. Gaussian Filtering

A manually implemented Gaussian filter is used for image smoothing.

### 6. Image Sharpening

Sharpening is applied to enhance facial image details.

### 7. Normalization

Images are prepared as normalized numerical arrays for CNN processing.

## CNN Model Architecture

The model uses a custom CNN architecture with four convolutional blocks.

### Architecture

```text
Input Image (128 × 128 × 1)
              |
              v
      Conv2D (32 Filters)
      Batch Normalization
         Max Pooling
              |
              v
      Conv2D (64 Filters)
      Batch Normalization
         Max Pooling
              |
              v
      Conv2D (128 Filters)
      Batch Normalization
         Max Pooling
              |
              v
      Conv2D (256 Filters)
      Batch Normalization
         Max Pooling
              |
              v
      Global Average Pooling
              |
              v
      Dense (256, ReLU)
              |
              v
         Dropout (0.5)
              |
              v
      Dense (7, Softmax)
              |
              v
      Facial Expression
        Classification
```

### Architecture Components

- Convolutional layers for feature extraction
- Batch Normalization for training stability
- Max Pooling for spatial dimensionality reduction
- Global Average Pooling for feature aggregation
- Dense layers for classification
- Dropout for regularization
- Softmax for multi-class probability prediction

## Model Training

The CNN is developed and trained using TensorFlow and Keras.

### Training Configuration

| Parameter | Configuration |
|-----------|---------------|
| Model | Custom CNN |
| Framework | TensorFlow / Keras |
| Input Shape | 128 × 128 × 1 |
| Output Classes | 7 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Batch Size | 64 |
| Maximum Epochs | 35 |
| Dropout Rate | 0.5 |

### Training Techniques

The training process incorporates:

- Data augmentation
- Batch Normalization
- Dropout regularization
- Early stopping
- Model checkpointing
- Learning rate scheduling

### Data Augmentation

The project uses image augmentation techniques such as:

- Rotation
- Horizontal and vertical shifts
- Shearing
- Zooming
- Horizontal flipping

These transformations introduce image variations during training.

## Model Evaluation

The trained CNN model was evaluated using the FER2013 test dataset.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training and Validation Curves

### Overall Performance

| Metric | Result |
|--------|--------|
| Test Accuracy | Approximately 58% |
| Weighted Precision | 0.55 |
| Weighted Recall | 0.58 |
| Weighted F1-score | 0.55 |
| Test Samples | 7,178 |

The evaluation provides an overview of the model's classification performance on unseen test images.

### Performance Visualization

The notebook includes visualizations of:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss
- Confusion matrix

## Custom Image Prediction

The project supports testing the trained model using custom images.

### Prediction Workflow

```text
Input Image
     |
     v
Face Detection
     |
     v
Image Preprocessing
     |
     v
Resize and Normalize
     |
     v
CNN Model
     |
     v
Softmax Probabilities
     |
     v
Predicted Expression
```

### Prediction Output

The notebook can display:

- Original image
- Preprocessed image
- Predicted facial expression
- Prediction confidence
- Probability distribution across classes

Predictions represent visible facial expression patterns and should not be interpreted as definitive measurements of a person's emotional state.

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN development and training |
| OpenCV | Face detection and image processing |
| NumPy | Numerical computations |
| Pandas | Data handling |
| Scikit-learn | Data preparation and evaluation |
| Matplotlib | Visualization |
| KaggleHub | Dataset acquisition |
| Google Colab | Development environment |
| Jupyter Notebook | Interactive experimentation |

## Installation and Setup

### Prerequisites

- Python 3.x
- Google Colab or Jupyter Notebook
- Required Python dependencies
- Internet connection for dataset downloading

Google Colab is recommended because the notebook uses Colab-specific functionality.

### 1. Clone the Repository

```bash
git clone https://github.com/Belalzz20/Facial-Emotion-Recognition-Using-Deep-Learning.git
cd Facial-Emotion-Recognition-Using-Deep-Learning
```

### 2. Install Dependencies

```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn kagglehub
```

### 3. Open the Notebook

Open:

`D_image.ipynb`

in Google Colab.

### 4. Download the Dataset

Execute the dataset-loading cells to download FER2013 using KaggleHub.

### 5. Preprocess the Images

Run the preprocessing pipeline to prepare the facial images for CNN training.

### 6. Train the Model

Execute the model-building and training cells in sequence.

### 7. Evaluate the Model

Run the evaluation cells to generate classification metrics and visualizations.

### 8. Test Custom Images

Use the custom image prediction functionality to upload and classify facial images.

**Note:** The interactive menu's training option references a function that is not defined in the notebook. Run the training cells directly instead.

## Project Structure

```text
Facial-Emotion-Recognition-Using-Deep-Learning/
|
|-- D_image.ipynb
|-- Facial Emotion Recognition.pdf
|-- References.pdf
|-- Survey of  Facial Emotion Recognition.pdf
|-- Test_1.jpg
|-- README.md
```

### File Descriptions

| File | Description |
|------|-------------|
| `D_image.ipynb` | CNN development, training, evaluation, and inference |
| `Facial Emotion Recognition.pdf` | Project documentation |
| `References.pdf` | Supporting references |
| `Survey of  Facial Emotion Recognition.pdf` | Related research survey |
| `Test_1.jpg` | Sample image for testing |
| `README.md` | Project overview and documentation |

## Project Workflow

1. Dataset acquisition
2. Image preprocessing
3. Face detection
4. Data preparation
5. CNN model development
6. Data augmentation
7. Model training
8. Model evaluation
9. Custom image prediction
10. Result visualization

## Future Improvements

- Explore transfer learning architectures
- Improve face detection robustness
- Optimize CNN hyperparameters
- Improve handling of class imbalance
- Develop a standalone inference application
- Add real-time webcam-based classification
- Improve experiment reproducibility
- Develop a web-based interface

## Learning Outcomes

This project demonstrates practical experience with:

- Deep Learning
- Computer Vision
- Convolutional Neural Networks
- Image preprocessing and enhancement
- Multi-class classification
- Data augmentation
- Model training and evaluation
- TensorFlow and Keras
- Model inference and visualization

## Project Information

**Project Name:** Facial Emotion Recognition Using Deep Learning

**Project Type:** Deep Learning / Computer Vision

**Programming Language:** Python

**Framework:** TensorFlow / Keras

**Dataset:** FER2013

**Development Environment:** Google Colab / Jupyter Notebook

## License

No license has been specified for this project.
