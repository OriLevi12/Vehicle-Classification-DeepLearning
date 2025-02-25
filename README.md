# Vehicle Classification with Deep Learning

## Overview
This repository contains a deep learning-based vehicle classification model. The project implements both a **Custom ResNet50 Model** and a **Baseline CNN Model**, comparing their performance in classifying different types of vehicles.

## Dataset
The dataset used for training and evaluation can be found on Kaggle:  
[Vehicle Type Image Dataset](https://www.kaggle.com/datasets/sujaykapadnis/vehicle-type-image-dataset)

The dataset consists of images of different vehicle types, which are preprocessed and cleaned before training.

## Features
- **Custom ResNet50 Model** with Transfer Learning
- **Baseline CNN Model** for comparison
- **Hyperparameter tuning** to find optimal learning rate, batch size, and dropout rate
- **Data augmentation** for improved generalization
- **Early stopping** to prevent overfitting
- **Performance evaluation** with confusion matrices and classification reports

## Installation & Setup

### 1. Clone the repository:
```bash
git clone https://github.com/OriLevi12/Vehicle-Classification-DeepLearning.git
cd Vehicle-Classification-DeepLearning
```

### 2. Environment & Dependencies:
This project is designed to run in **Google Colab**. Most dependencies are pre-installed, but if running locally, install the required libraries:

```bash
pip install torch torchvision matplotlib seaborn scikit-learn
```

### 3. Download the dataset:
Since the dataset is not included in the repository, download it manually from **Kaggle** and place it in your **Google Drive** under:

```bash
My Drive/data/vehicles_dataset.zip
```

**Note:** The dataset is extracted automatically within the notebook.

### 4. Run the Jupyter Notebook in Google Colab
Upload the notebook to **Google Colab**, ensure your **Google Drive** is mounted, and execute the cells step by step.

## Training the Model
1. The dataset is extracted and validated for corrupted images.
2. **Hyperparameter tuning** is performed on ResNet50 to find the optimal learning rate, batch size, and dropout rate.
3. The **Baseline CNN Model** and **Custom ResNet50 Model** are trained separately.
4. The trained models are evaluated using accuracy, confusion matrices, and classification reports.
5. Results are visualized with training loss, validation loss, and accuracy graphs.

## Model Evaluation
The trained models are evaluated using:
- **Confusion Matrix**: Visual representation of misclassifications
- **Classification Report**: Precision, Recall, and F1-score for each class
- **Accuracy Comparison**: Between Custom ResNet50 and Baseline CNN

## Results & Findings
- The **Custom ResNet50 model** achieves significantly higher accuracy compared to the Baseline CNN due to the benefits of transfer learning.
- **Data augmentation** helps improve generalization and performance.
- The best-performing model is saved and can be loaded for inference.

## 📬 Contact Info
**Ori Levi**  
📧 Email: Leviori1218@gmail.com  
🐙 GitHub: [OriLevi12](https://github.com/OriLevi12)
