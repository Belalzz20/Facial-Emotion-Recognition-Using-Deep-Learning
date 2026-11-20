
# Facial Emotion Recognition Using Deep Learning

### Deep Learning | Computer Vision | CNN | TensorFlow | Keras | OpenCV

## Overview

This project implements a facial expression recognition system using Deep Learning and Computer Vision techniques.

A custom Convolutional Neural Network (CNN) is trained on the FER2013 dataset to classify facial expressions into seven categories.

The project covers the machine learning workflow, including dataset preparation, image preprocessing, data augmentation, CNN training, model evaluation, and predictions on custom images.

The system also provides visualizations of predicted facial expressions and confidence scores.

## Key Features

- Facial expression classification using a CNN
- Seven facial expression categories
- Face detection using OpenCV Haar Cascades
- Image preprocessing and enhancement
- CLAHE-based contrast enhancement
- Gaussian filtering and image sharpening
- Data augmentation for model training
- CNN architecture with Batch Normalization and Dropout
- Training and validation performance visualization
- Classification report and confusion matrix
- Custom image predictions
- Prediction confidence visualization

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

The project uses the **FER2013 (Facial Expression Recognition 2013)** dataset, a benchmark dataset for facial expression classification.

The dataset contains grayscale facial images organized into training and testing directories.

**Dataset Source:** [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

The dataset is downloaded using KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download("msambare/fer2013")
```

### Dataset Preparation

The dataset preparation process includes:

- Loading training and testing images
- Converting images to grayscale
- Detecting and cropping facial regions
- Applying image enhancement
- Resizing images to 128 × 128 pixels
- Preparing labels for multi-class classification
- Creating training and validation data

## Image Preprocessing

Image preprocessing is applied before passing images to the CNN model.

### 1. Grayscale Conversion

Input images are converted into grayscale to simplify image representation and reduce computational complexity.

### 2. Face Detection

OpenCV Haar Cascade classifiers are used to detect facial regions.

When a face is detected, the first detected face is cropped for further processing.

### 3. Image Resizing

Processed images are resized to a consistent resolution of **128 × 128 pixels**.

### 4. Contrast Enhancement

Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied to improve local image contrast.

### 5. Gaussian Filtering

A manually implemented Gaussian filter is used for image smoothing.

### 6. Image Sharpening

A sharpening filter is applied to enhance image details.

### 7. Image Normalization

Processed images are normalized and prepared as numerical arrays for CNN inference.

## CNN Model Architecture

The project uses a custom Convolutional Neural Network designed for multi-class facial expression classification.

The architecture consists of four convolutional blocks followed by fully connected classification layers.

### Architecture Overview

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

- **Convolutional Layers:** Extract facial image features.
- **Batch Normalization:** Supports training stability.
- **Max Pooling:** Reduces spatial dimensions.
- **Global Average Pooling:** Aggregates extracted features.
- **Dense Layers:** Perform classification.
- **Dropout:** Provides regularization.
- **Softmax:** Produces class probability scores.

## Model Training

The CNN model is developed and trained using TensorFlow and Keras.

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

The training workflow incorporates:

- Data augmentation
- Batch Normalization
- Dropout regularization
- Early stopping
- Model checkpointing
- Learning rate scheduling

### Data Augmentation

The project uses augmentation techniques including:

- Rotation
- Horizontal and vertical shifts
- Shearing
- Zooming
- Horizontal flipping

These transformations introduce variations during training.

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

The notebook includes visualizations for:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss
- Confusion matrix

## Model Predictions and Results

The following screenshots demonstrate the CNN model's facial expression classification and image preprocessing capabilities.

### 1. FER2013 Evaluation Samples

Examples of facial expression predictions from the FER2013 evaluation dataset.

The screenshots show the ground-truth labels alongside the predicted labels.

![FER2013 Evaluation Samples](screenshots/evaluation-samples.png)

### 2. Custom Image Prediction - Ronaldo

This example demonstrates the model's prediction workflow on a custom image.

The image is processed to extract the facial region and prepare it for CNN inference.

**Prediction Results:**

- **Predicted Expression:** Happy
- **Confidence Score:** 99.99%
- **Processed Image Size:** 128 × 128 pixels

![Ronaldo Prediction](screenshots/ronaldo-prediction.png)

### 3. Custom Image Prediction

Another example demonstrates facial image preprocessing and classification.

The original image is transformed into a processed grayscale facial image before being passed to the CNN.

**Prediction Results:**

- **Predicted Expression:** Happy
- **Confidence Score:** 94.45%
- **Processed Image Size:** 128 × 128 pixels

![Custom Image Prediction](screenshots/custom-prediction.png)

**Note:** Confidence scores represent model output probabilities and do not guarantee classification accuracy. Facial expression predictions should not be interpreted as definitive measurements of a person's emotional state.

## Custom Image Prediction Workflow

The project supports testing the trained model using custom images.

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
     Predicted Expression
             |
             v
    Prediction Visualization
```

### Prediction Output

The prediction functionality displays:

- Original input image
- Processed facial image
- Predicted facial expression
- Prediction confidence score
- Class probability distribution

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN development and training |
| OpenCV | Face detection and image processing |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Scikit-learn | Data preparation and model evaluation |
| Matplotlib | Data visualization |
| KaggleHub | Dataset downloading |
| Google Colab | Development and execution |
| Jupyter Notebook | Interactive experimentation |

## Installation and Setup

### Prerequisites

- Python 3.x
- Google Colab or Jupyter Notebook
- Required Python libraries
- Internet connection for dataset downloading

Google Colab is recommended because the notebook includes Colab-specific functionality.

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

Open the following notebook in Google Colab:

```text
D_image.ipynb
```

### 4. Download the Dataset

Execute the dataset-loading cells to download FER2013 using KaggleHub.

### 5. Preprocess the Images

Run the preprocessing cells to prepare the images for CNN training.

### 6. Train the CNN Model

Execute the model-building and training cells in sequence.

The notebook includes model checkpointing and model saving functionality.

### 7. Evaluate the Model

Run the evaluation cells to generate:

- Classification report
- Confusion matrix
- Training and validation visualizations

### 8. Test Custom Images

Use the custom image prediction functionality to upload and classify facial images.

**Execution Note:** The notebook's interactive menu references a training function that is not defined. Run the existing training cells directly.

## Project Structure

```text
Facial-Emotion-Recognition-Using-Deep-Learning/
|
|-- D_image.ipynb
|-- Facial Emotion Recognition.pdf
|-- References.pdf
|-- Survey of Facial Emotion Recognition.pdf
|-- Test_1.jpg
|-- README.md
|
|-- screenshots/
    |-- evaluation-samples.png
    |-- ronaldo-prediction.png
    |-- custom-prediction.png
```

### File Descriptions

| File | Description |
|------|-------------|
| `D_image.ipynb` | CNN development, training, evaluation, and inference |
| `Facial Emotion Recognition.pdf` | Project documentation |
| `References.pdf` | Supporting references |
| `Survey of Facial Emotion Recognition.pdf` | Related research survey |
| `Test_1.jpg` | Sample image for testing |
| `screenshots/` | Prediction and evaluation screenshots |
| `README.md` | Project overview and documentation |

## Project Workflow

The overall project workflow consists of:

1. Dataset acquisition
2. Image preprocessing
3. Face detection
4. Data preparation
5. CNN architecture development
6. Data augmentation
7. Model training
8. Model evaluation
9. Custom image prediction
10. Result visualization

## Future Improvements

Potential future developments include:

- Exploring transfer learning architectures
- Improving facial detection robustness
- Optimizing CNN hyperparameters
- Improving class imbalance handling
- Developing a standalone inference application
- Supporting real-time webcam-based classification
- Adding automated model testing
- Improving experiment reproducibility
- Developing a web-based interface

## Learning Outcomes

This project demonstrates practical experience with:

- Deep Learning
- Computer Vision
- Convolutional Neural Networks
- Image preprocessing and enhancement
- Face detection
- Multi-class classification
- Data augmentation
- Model training and evaluation
- TensorFlow and Keras development
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
