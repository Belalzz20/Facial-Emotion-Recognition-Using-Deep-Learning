
# Facial Emotion Recognition Using Deep Learning

### Deep Learning | Computer Vision | CNN | TensorFlow | OpenCV

## Overview

This project implements a Facial Emotion Recognition system using Deep Learning and Computer Vision techniques.

A Convolutional Neural Network (CNN) is trained on the FER2013 dataset to classify facial expressions into seven emotion categories.

The project combines image preprocessing, face detection, data augmentation, CNN-based classification, and model evaluation to build an end-to-end facial expression recognition workflow.

The system also supports testing custom images and visualizing predicted emotion categories with confidence scores.

## Key Features

- Facial expression classification using CNN.
- Seven-class emotion recognition.
- Face detection using OpenCV Haar Cascades.
- Image preprocessing and enhancement.
- CLAHE-based contrast enhancement.
- Manual Gaussian filtering and image sharpening.
- Data augmentation for model training.
- CNN architecture with Batch Normalization and Dropout.
- Model evaluation using classification metrics.
- Confusion matrix visualization.
- Training and validation performance visualization.
- Custom image prediction and confidence visualization.

## Emotion Categories

The model classifies facial expressions into seven categories:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

## Dataset

### FER2013 - Facial Expression Recognition Dataset

The project uses the FER2013 dataset, a widely used benchmark for facial expression classification.

The dataset contains grayscale facial images organized into training and testing directories.

The dataset is downloaded using KaggleHub.

```python
import kagglehub

path = kagglehub.dataset_download("msambare/fer2013")
```

**Dataset Source:**

https://www.kaggle.com/datasets/msambare/fer2013

### Dataset Preparation

The dataset preparation process includes:

- Loading images from training and testing directories.
- Converting images to grayscale.
- Applying face detection and preprocessing.
- Resizing images to 128 × 128 pixels.
- Preparing labels for multi-class classification.
- Creating training, validation, and testing data.

## Image Preprocessing

Image preprocessing is an important part of the project.

The implemented pipeline includes:

### 1. Grayscale Conversion

Input images are converted into grayscale to simplify image representation and reduce computational complexity.

### 2. Face Detection

OpenCV Haar Cascade classifiers are used to detect faces.

When a face is detected, the first detected face is cropped for further processing.

### 3. Image Resizing

Processed images are resized to:

**128 × 128 pixels**

This provides a consistent input shape for the CNN model.

### 4. Contrast Enhancement

Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied to improve local image contrast.

### 5. Gaussian Filtering

A manually implemented Gaussian filter is used for image smoothing.

### 6. Image Sharpening

A sharpening filter is applied to enhance facial image details.

### 7. Image Standardization

Processed images are prepared for CNN input using consistent pixel normalization and tensor dimensions.

## CNN Model Architecture

The project uses a custom Convolutional Neural Network designed for multi-class facial expression classification.

The architecture includes:

- Convolutional layers.
- Batch Normalization.
- Max Pooling.
- Global Average Pooling.
- Fully Connected layers.
- Dropout regularization.
- Softmax classification.

### Architecture Overview

```text
Input Image
(128 × 128 × 1)
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
Dense Layer (256)
ReLU Activation
        |
        v
Dropout (0.5)
        |
        v
Dense Layer (7)
Softmax Activation
        |
        v
Predicted Emotion
```

## Model Training

The CNN is trained using TensorFlow and Keras.

### Training Configuration

| Parameter | Configuration |
|-----------|---------------|
| Model | Convolutional Neural Network |
| Framework | TensorFlow / Keras |
| Input Size | 128 × 128 × 1 |
| Output Classes | 7 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Batch Size | 64 |
| Maximum Epochs | 35 |
| Dropout Rate | 0.5 |

### Training Techniques

The training workflow includes:

- Data augmentation.
- Batch Normalization.
- Dropout regularization.
- Early stopping.
- Model checkpointing.
- Learning rate scheduling.

### Data Augmentation

Image augmentation is used to introduce variations during training.

The augmentation techniques include:

- Rotation.
- Width and height shifts.
- Shearing.
- Zooming.
- Horizontal flipping.

These transformations help the model learn from variations in facial image appearance.

## Model Evaluation

The trained CNN model was evaluated on the FER2013 test dataset using standard multi-class classification metrics.

### Evaluation Metrics

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion Matrix.
- Training and Validation Curves.

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

The notebook includes visualizations for:

- Training accuracy.
- Validation accuracy.
- Training loss.
- Validation loss.
- Confusion matrix.

These visualizations support analysis of the model's learning behavior and classification results.

## Emotion Prediction

The project includes functionality for testing the trained model on custom images.

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
Resize to 128 × 128
     |
     v
CNN Model
     |
     v
Softmax Probabilities
     |
     v
Predicted Emotion
     |
     v
Prediction Visualization
```

### Prediction Output

The prediction workflow can display:

- Original input image.
- Preprocessed facial image.
- Predicted emotion category.
- Prediction confidence score.
- Class probability distribution.

The predictions represent facial expression categories inferred from images, not a definitive assessment of a person's internal emotional state.

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development |
| OpenCV | Image processing and face detection |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Scikit-learn | Model evaluation and data preparation |
| Matplotlib | Data visualization |
| KaggleHub | Dataset downloading |
| Google Colab | Model development and execution |
| Jupyter Notebook | Interactive development |

## Installation and Setup

### Prerequisites

- Python 3.x
- Jupyter Notebook or Google Colab
- Required Python libraries
- Internet connection for downloading the dataset

Google Colab is recommended because the notebook includes Colab-specific image upload and download functionality.

### 1. Clone the Repository

```bash
git clone https://github.com/Belalzz20/Facial-Emotion-Recognition-CNN.git

cd Facial-Emotion-Recognition-CNN
```

### 2. Install Dependencies

```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn kagglehub
```

### 3. Open the Notebook

Open the following file in Google Colab or Jupyter Notebook:

```text
D_image.ipynb
```

### 4. Download the Dataset

Execute the dataset-loading cells in the notebook.

The FER2013 dataset will be downloaded using KaggleHub.

### 5. Run the Preprocessing Pipeline

Execute the preprocessing cells to prepare facial images for training and testing.

### 6. Train the CNN Model

Run the model-building and training cells in sequence.

The notebook includes model checkpointing and saves trained model artifacts.

### 7. Evaluate the Model

Execute the evaluation cells to generate:

- Classification report.
- Confusion matrix.
- Training and validation visualizations.

### 8. Test Custom Images

Run the image prediction cells to upload and classify facial images.

**Note:** The notebook's interactive menu contains a training option that references an undefined function. Training should be performed through the existing training cells.

## Project Structure

```text
Facial-Emotion-Recognition-CNN/
|
|-- D_image.ipynb
|-- README.md
|
|-- docs/
|   |-- Facial Emotion Recognition.pdf
|   |-- Survey of Facial Emotion Recognition.pdf
|   |-- References.pdf
|
|-- sample_images/
    |-- Ronaldo 1.jpg
    |-- ronaldo 2.jpg
    |-- ronaldo3.jpg
    |-- Test_1.jpg
```

This is the recommended GitHub repository structure.

Sample images should only be included when redistribution is permitted.

## Project Workflow

The overall development workflow consists of:

1. Dataset acquisition.
2. Image preprocessing.
3. Face detection.
4. Data preparation.
5. CNN architecture development.
6. Data augmentation.
7. Model training.
8. Model evaluation.
9. Emotion prediction.
10. Result visualization.

## Future Improvements

Potential future developments include:

- Exploring transfer learning architectures.
- Improving facial detection robustness.
- Optimizing CNN architecture and hyperparameters.
- Improving class imbalance handling.
- Developing a standalone inference application.
- Supporting real-time webcam-based prediction.
- Adding automated model testing.
- Improving model reproducibility.
- Developing a web-based user interface.

## Learning Outcomes

This project demonstrates practical experience in:

- Deep Learning.
- Computer Vision.
- Convolutional Neural Networks.
- Image preprocessing and enhancement.
- Face detection.
- Multi-class image classification.
- Data augmentation.
- Model training and evaluation.
- TensorFlow and Keras development.
- Model inference and visualization.

## Project Information

**Project Name:** Facial Emotion Recognition Using Deep Learning

**Project Type:** Deep Learning / Computer Vision

**Programming Language:** Python

**Deep Learning Framework:** TensorFlow / Keras

**Dataset:** FER2013

**Development Environment:** Google Colab / Jupyter Notebook

## License

No license has been specified for this project.
