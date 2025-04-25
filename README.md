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

  # Dataset Preparation and Augmentation

This project utilizes dermoscopic images from the **ISIC 2024 Challenge Dataset** to construct a high-quality training and validation set for model development.

## Dataset Selection

- **Training Set**:
  - 40,000 negative samples (class 0)
  - 1,672 positive samples (class 1)

- **Validation Set**:
  - 10,000 negative samples (class 0)
  - 419 positive samples (class 1)

Images are resized to **224x224** pixels and normalized using ImageNet mean and standard deviation:
- Mean: `[0.485, 0.456, 0.406]`
- Standard Deviation: `[0.229, 0.224, 0.225]`

To address class imbalance, random shuffling and balanced sampling strategies are applied during training.

## Data Augmentation

During training, the following data augmentations are applied probabilistically to enhance model generalization:

- **Random Horizontal Flip**: Probability = 0.5
- **Random Rotation**: Range = [-30°, +30°]
- **Random Scaling**: Scale factor between 0.8 and 1.2
- **Color Jitter**:
  - Brightness adjustment: ±0.2
  - Contrast adjustment: ±0.2
  - Saturation adjustment: ±0.2
  - Hue adjustment: ±0.2

## Preprocessing

Prior to augmentation, all images undergo:
- **Hair Artifact Removal** using a black-hat morphological operation and inpainting (Telea method)
- **Perceptual Filtering** using VGG16 feature extraction to retain high-quality generated samples

---

This preprocessing and augmentation pipeline ensures that the training dataset is both diverse and clinically realistic, which is critical for robust model performance.
