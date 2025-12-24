# Artwork Style & Artist Classification using Deep Learning

An end-to-end AI project that predicts the **art style** and **artist** of an artwork image using **Deep Learning with PyTorch**.  
The system supports image upload through a **GUI**, provides predictions, and explains model decisions using **Grad-CAM**.

---

## Project Overview

This project aims to build a real-world **Computer Vision** application that:
- Classifies artwork images by **Style** and **Artist**
- Handles a **large-scale dataset (~33GB)**
- Compares **3 different CNN-based models**
- Provides **model explainability** using Grad-CAM
- Offers a **user-friendly GUI** for easy interaction

---

## Key Features

✔ Trained **3 Deep Learning models** using PyTorch  
✔ Large dataset preprocessing & handling (~33GB)  
✔ GUI for image upload & real-time prediction  
✔ Model evaluation (Accuracy, Confusion Matrix, etc.)  
✔ Explainable AI using **Grad-CAM heatmaps**  

---

## Tech Stack

- **Python**
- **PyTorch**
- **Torchvision**
- **OpenCV**
- **NumPy & Pandas**
- **Matplotlib**
- **Grad-CAM**
- **GUI Framework (streamlit)**

---

## Dataset

- Size: **~33 GB**
- Type: Artwork images
- Classes:
  - Multiple **Art Styles**
  - Multiple **Artists**

> Due to size limitations, the dataset is not uploaded to GitHub.
- link of dataset-->
- https://www.kaggle.com/datasets/steubk/wikiart
---

## Models Used

We implemented and evaluated **3 CNN-based models (efficientb0, resnet50, vgg16)**, comparing their performance in terms of:
- Accuracy
- Generalization
- Interpretability

---

## Evaluation

Evaluation techniques used:
- Accuracy
- Precision & Recall
- Confusion Matrix
- Visual interpretation with **Grad-CAM**

---

## Explainability (Grad-CAM)

Grad-CAM was used to visualize the important regions in the artwork that influenced the model’s predictions, improving:
- Transparency
- Trustworthiness
- Model understanding

---

## GUI Demo

The GUI allows users to:
1. Upload an artwork image
2. Predict its **Style**
3. Visualize Grad-CAM results
---


Project structure ready.
