# 🔫 Gun Detection with YOLOv8

This repository contains my experiments on training a **gun detection model** using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).  
The dataset used is from [Kaggle – Guns Object Detection](https://www.kaggle.com/datasets/issaisasank/guns-object-detection).  

---

### Project Overview
The main goal of this project is to build a **real-time camera surveillance system** that can detect guns for safety and security purposes.  

Steps I followed:
- Converted the original label format to YOLO format .  
- Split the dataset into training and validation sets (80/20).  
- Validated labels and checked for missing annotations.  
- Trained YOLOv8 with different hyperparameters.  

---

###  Results So Far
- **Best result achieved:**  
  - Precision/Recall gave a **mAP50-95 of ~0.63 (63%)** using the `best.pt` model from previous training.  

- **Issues:**  
  - Some experiments resulted in poor performance (very low precision/recall, NaN loss values).  
  - Increasing epochs didn’t always improve results ,sometimes the model diverged.  
  - The dataset is small (~333 images), which limits generalization.  

⚠ **Note:** The results are **not very good yet**. This repository is a **work-in-progress**, and I will keep updating it with better experiments, improved preprocessing, and additional data augmentation.  

---

### ⚙ Usage

### Install dependencies
```bash
pip install ultralytics opencv-python numpy
