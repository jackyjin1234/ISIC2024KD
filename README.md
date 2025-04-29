# ISIC2024KD

# Feature Matching Knowledge Distillation (feature_matching.ipynb)

The distillation strategy combines two objectives:

- **Classification Loss**:  
  Standard cross-entropy loss between the student’s predictions and the ground-truth labels.
  
- **Feature Matching Loss**:  
  Mean Squared Error (MSE) loss between intermediate feature maps of the teacher and the student.  
  Specific layers (e.g., after certain blocks) from both models are selected and aligned in size to compute the feature loss.

The total loss is a weighted sum:

$$
\text{Total Loss} = (1 - \alpha) \times \text{Classification Loss} + \alpha \times \text{Feature Matching Loss}
$$

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

# Response-based Knowledge Distillation (distill_reponse.py)

This approach focuses on **matching the softened output logits** of the teacher and student networks.  
Specifically, it combines:
- A weighted **Cross-Entropy Loss** on the student’s predictions, and
- A **KL Divergence Loss** between the softened outputs (logits) of teacher and student.

##  Hyperparameters

| Parameter               | Value                             | Description                                           |
|--------------------------|-----------------------------------|-------------------------------------------------------|
| `T` (Temperature)        | 4.0                               | Softens logits to expose dark knowledge               |
| `α` (Alpha)              | 0.6                               | Weight between CE loss (α) and KL loss (1-α)          |
| Batch size               | 32                                | Images per batch                                     |
| Image size               | 256×256 (teacher) / 224×224 (student) | Input resizing strategy                          |
| Learning rate            | 1e-4                              | Adam optimizer                                       |
| Optimizer                | Adam                              | Adaptive Moment Estimation                           |
| Scheduler                | ReduceLROnPlateau                 | Reduce learning rate if validation pAUC plateaus     |
| Epochs                   | 20                                | Number of training epochs                           |
| Save checkpoint          | Best model based on validation pAUC | Best model is automatically saved                 |
| `n_estimators`           | 600                                       | Number of boosting rounds (trees)           |
| `learning_rate`          | 0.05                                      | Shrinkage (step size)                       |
| `max_depth`              | 5                                         | Maximum depth of each tree                  |
| `subsample`              | 0.8                                       | Fraction of rows sampled per tree           |
| `colsample_bytree`       | 0.8                                       | Fraction of features sampled per tree       |
| `scale_pos_weight`       | `sum(y_tr==0)/sum(y_tr==1)` (≈ 24)        | Balances class imbalance                    |
| `eval_metric`            | `auc`                                     | Evaluation metric for early stopping       |
| `tree_method`            | `gpu_hist` (or `hist` if no GPU)         | Tree construction algorithm                 |
| `early_stopping_rounds`  | 50                                        | Stop if validation AUC does not improve     |
| `random_state`           | 42                                        | Seed for reproducible train/validation split|

**Weighted Loss:**  
To address severe class imbalance in skin cancer data, a **custom class-weighted CrossEntropyLoss** is used, emphasizing the malignant class.

# Token-based Knowledge Distillation (TokenKD.py)
Token-based knowledge distillation aims to transfer fine-grained knowledge from a large teacher model to a smaller student model by matching their outputs and internal representations at the token level, rather than just at the final prediction or feature level. This approach enables the student model to better capture the nuanced information encoded in the teacher’s token-wise outputs, leading to improved performance.

### Distillation Loss

The total loss used for training is a weighted sum of three components:

1. **Classification Loss:**
   Standard cross-entropy loss between the student’s predictions and the ground-truth labels, with class weighting to address imbalance.
2. **Token-level KL Divergence Loss:**
   KL divergence between the softened output logits (using temperature scaling) of the teacher and student models. This encourages the student to mimic the teacher’s token-wise output distribution, capturing “dark knowledge” about class relationships.
3. **Feature Matching Loss:**
   Mean squared error (MSE) between the intermediate token features of the teacher and student. This enforces alignment not just at the output, but throughout the token representations.

The combined loss is:

$$
\text{Total Loss} = (1 - \alpha) \times \text{Classification Loss} + \alpha \times (0.7 \times \text{KL Loss} + 0.3 \times \text{Feature Loss})
$$

where $\alpha$ balances supervised and distillation objectives, and the 0.7/0.3 split weights the importance of output versus feature alignment.

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
