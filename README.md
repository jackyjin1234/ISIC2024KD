# ISIC2024KD

# Feature Matching Knowledge Distillation

This project implements a **feature matching knowledge distillation** framework where a lightweight **ViT-small** student model is trained to mimic intermediate feature representations of a **SwinV2** teacher model.  
The task focuses on binary classification using the **ISIC 2024** Challenge medical imaging dataset, with a highly imbalanced distribution (40,000 benign vs. 1,672 malignant samples).  
The training pipeline applies standard data augmentations (random cropping, flipping, rotation), and optimizes a combined classification loss and feature-level MSE loss.  
Model performance is evaluated using **AUC**, **PR-AUC**, and **pAUC** metrics.

## Distillation Strategy

The distillation strategy combines two objectives:

- **Classification Loss**:  
  Standard cross-entropy loss between the student’s predictions and the ground-truth labels.
  
- **Feature Matching Loss**:  
  Mean Squared Error (MSE) loss between intermediate feature maps of the teacher and the student.  
  Specific layers (e.g., after certain blocks) from both models are selected and aligned in size to compute the feature loss.

The total loss is a weighted sum:

\[
\text{Total Loss} = (1 - \alpha) \times \text{Classification Loss} + \alpha \times \text{Feature Matching Loss}
\]

where \(\alpha\) controls the balance between the two parts.

## Code Implementation Overview

- **Teacher and Student Models**:  
  Load a pre-trained SwinV2 teacher and initialize a ViT-small student model from `timm`.

- **Feature Extraction**:  
  Modify both models to expose intermediate features during the forward pass.  
  The student is trained not only to predict labels but also to match the selected teacher features.

- **Training Loop**:  
  In each iteration:
  1. Forward both teacher and student on the same input.
  2. Compute classification loss using student's output.
  3. Compute feature matching loss (MSE) between teacher and student features.
  4. Combine the two losses and perform backpropagation.

- **Evaluation**:  
  After each epoch, evaluate model performance using validation AUC, PR-AUC, and pAUC@0.8 metrics.

## Environment
- Python 3.8+
- PyTorch
- timm
- scikit-learn
- torchvision
- h5py
- tqdm
- PIL
- matplotlib
