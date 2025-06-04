# 🧠 Alzheimer's Disease Classification Using Deep Learning on MRI Scans

A deep learning-based pipeline for classifying Alzheimer’s Disease stages from MRI brain scans using **Custom CNNs** and **VGG16-based Transfer Learning**. This project explores the efficacy of deep neural networks in diagnosing AD, focusing on preprocessing, model architecture, and performance comparison across two medical imaging datasets.

---

## 📖 Overview

Alzheimer’s Disease (AD) is a neurodegenerative condition causing cognitive decline, for which early detection is crucial. In this study, we leverage **Convolutional Neural Networks (CNN)** and **Transfer Learning** (VGG16) to build robust models capable of detecting and classifying stages of Alzheimer’s Disease using brain MRI images.

---

## 📚 Datasets

### 🧬 1. ADNI Dataset (Alzheimer’s Disease Neuroimaging Initiative)

- **Classes**: CN (Cognitively Normal), EMCI, LMCI, AD  
- **Images**: 39,929 total  
- **Image Format**: Preprocessed MRI scans  
- **Standardized Size**: 224×224 pixels

### 🧠 2. Alzheimer's Disease Multiclass Dataset

- **Classes**: NonDemented, VeryMildDemented, MildDemented, ModerateDemented  
- **Images**: 44,000 total  
- **Image Format**: JPEG, skull-stripped and cleaned  
- **Standardized Size**: 224×224 pixels

---

## ⚙️ Preprocessing & Image Enhancement

- **Resizing & Normalization**
- **Data Augmentation** (rotation, zoom, shift, flip)
- **Edge Detection**: Canny, Laplacian, Sobel
- **Segmentation**: Binary thresholding and K-Means clustering
- **Contrast Enhancement**: Histogram Equalization & CLAHE

These techniques enhanced the model's ability to detect structural anomalies such as fluid-filled regions and gray/white matter boundaries.

---

## 🏗️ Model Architectures

### 🔸 VGG16 Transfer Learning Model
- Pre-trained on ImageNet (`include_top=False`)
- Frozen convolutional layers for feature extraction
- Custom fully connected layers: [2048, 1024]
- Batch Normalization & Softmax for classification

### 🔹 Custom CNN Model
- 4 Convolutional layers (filters: 32 → 64 → 128 → 256)
- ReLU activation + Batch Normalization
- MaxPooling layers to reduce dimensionality
- Dropout layers (0.25–0.5) to prevent overfitting
- Dense classifier + Softmax output

> All models trained with the **Adam optimizer**, **categorical crossentropy**, and stratified train/test splits.

---

## 📊 Experimental Results

| Model | Dataset | Task | Accuracy |
|-------|---------|------|----------|
| Model 1 (VGG16) | Multiclass | ModerateD vs ND | 98% |
| Model 1 (VGG16) | Multiclass | 4-class (ND, VeryMildD, MildD, ModerateD) | 98% |
| Model 2 (VGG16) | ADNI | AD vs CN | 98% |
| Model 2 (VGG16) | ADNI | 4-class (CN, EMCI, LMCI, AD) | 98% |
| Model 3 (Custom CNN) | ADNI | AD vs CN | 99% |
| Model 3 (Custom CNN) | ADNI | 4-class classification | 99% |
| Model 4 (Custom CNN) | Multiclass | ModerateD vs ND | 97% |
| Model 4 (Custom CNN) | Multiclass | 4-class classification | 98% |

---

## 📈 Performance Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Training/Validation Curve Analysis


