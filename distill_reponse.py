import os
import glob
import numpy as np
import pandas as pd
import pandas.api.types
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime
import csv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
IMAGE_SIZE = 256  # Changed from 224 to 256 to match model requirements

# -------------------------
# pAUC scoring function
# -------------------------
class ParticipantVisibleError(Exception):
    pass

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, min_tpr: float=0.80) -> float:
    """
    2024 ISIC Challenge metric: pAUC above a given TPR.
    """
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if not pandas.api.types.is_numeric_dtype(submission.values):
        raise ParticipantVisibleError('Submission target column must be numeric')

    # Get ground truth and predictions
    v_gt = np.asarray(solution.values)
    v_pred = np.asarray(submission.values)
    
    # If all samples are positive, return 1.0 (perfect score)
    if np.all(v_gt == 1):
        return 1.0
        
    max_fpr = abs(1 - min_tpr)

    fpr, tpr, _ = roc_curve(v_gt, v_pred)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError(f"Expected min_tpr in [0,1), got: {min_tpr}")

    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop-1], fpr[stop]]
    y_interp = [tpr[stop-1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    return auc(fpr, tpr)

# -------------------------
# Dataset Definition
# -------------------------
class SkinLesionDataset(Dataset):
    def __init__(self, root_dir, transform=None, neg_limit=40000, is_train=True):
        """
        Load images:
         - extra-train-image-1_hairRemove (label=1)
         - train-image-1_hairRemove (label=1)
         - train-image-0_hairRemove (label=0), use first neg_limit only
        """
        pos_dirs = ['extra-train-image-1_hairRemove', 'train-image-1_hairRemove']
        neg_dir = 'train-image-0_hairRemove'
        self.transform = transform
        self.paths, self.labels = [], []

        # Helper function to filter valid image files
        def is_image_file(path):
            return os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))

        # Load all positive samples
        pos_paths, pos_labels = [], []
        for d in pos_dirs:
            d_path = os.path.join(root_dir, d)
            if os.path.exists(d_path):
                for p in glob.glob(os.path.join(d_path, '*')):
                    if is_image_file(p):
                        pos_paths.append(p)
                        pos_labels.append(1)
        
        # Load negative samples (with limit)
        neg_paths, neg_labels = [], []
        neg_path = os.path.join(root_dir, neg_dir)
        if os.path.exists(neg_path):
            neg_files = [p for p in glob.glob(os.path.join(neg_path, '*')) if is_image_file(p)]
            neg_files = sorted(neg_files)[:neg_limit]
            for p in neg_files:
                neg_paths.append(p)
                neg_labels.append(0)

        all_paths = pos_paths + neg_paths
        all_labels = pos_labels + neg_labels

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_paths, all_labels,
            stratify=all_labels,
            test_size=0.1,
            random_state=42,
        )

        if is_train:
            self.paths = train_paths
            self.labels = train_labels
        else:
            self.paths = val_paths
            self.labels = val_labels

        logger.info(f"Loaded {'train' if is_train else 'val'} dataset with {len(self.paths)} images "
                    f"({sum(self.labels)} positive, {len(self.paths) - sum(self.labels)} negative)")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], os.path.basename(self.paths[idx])

def load_teacher(ckpt_path, device):
    """
    Load pretrained SwinV2_small teacher.
    """
    model = timm.create_model('swinv2_small_window8_256', 
                            pretrained=False, 
                            num_classes=2)
    model.load_state_dict(torch.load('clean_swin_weights.pth'))
    model.to(device)
    model.eval()
    return model

def load_student(device):
    """
    Create ViT small student.
    Using vit_small_patch16_224 but with interpolation in forward pass to handle 256x256 input.
    """
    model = timm.create_model('vit_small_patch16_224', pretrained=True)
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, 2)  # Binary classification
    model.to(device)
    return model

# -------------------------
# Distillation Losses
# -------------------------
def response_based_loss(s_logits, t_logits, labels, T=4.0, alpha=0.6):
    """
    Response-based knowledge distillation loss.
    Combines cross-entropy with KL divergence between softmax distributions.
    """
    # Debug info
    logger.debug(f"Response-based loss inputs:")
    logger.debug(f"- Student logits: {s_logits.shape}")
    logger.debug(f"- Teacher logits: {t_logits.shape}")
    logger.debug(f"- Labels: {labels.shape}")
    
    # Ensure same batch size
    if s_logits.size(0) != t_logits.size(0):
        raise ValueError(f"Batch size mismatch: student={s_logits.size(0)}, teacher={t_logits.size(0)}")
    a = torch.tensor([0.0625, 1.0]).to('cuda')
    ce = nn.CrossEntropyLoss(weight=a)(s_logits, labels)
    
    # Temperature scaling
    s_logits_temp = s_logits / T
    t_logits_temp = t_logits / T
    
    # KL divergence
    s_log_probs = nn.functional.log_softmax(s_logits_temp, dim=1)
    t_probs = nn.functional.softmax(t_logits_temp, dim=1)
    
    # Debug info
    logger.debug(f"- Student log probs: {s_log_probs.shape}")
    logger.debug(f"- Teacher probs: {t_probs.shape}")
    
    kd = nn.KLDivLoss(reduction='batchmean')(s_log_probs, t_probs) * (T * T)
    
    return alpha * ce + (1 - alpha) * kd

# -------------------------
# Training & Evaluation
# -------------------------
def train_one_epoch(student, teacher, loader, optimizer, device, method):
    student.train()
    loss_sum = 0.0
    pbar = tqdm(loader, desc='Training')
    
    for imgs, labels, _ in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        batch_size = imgs.size(0)
        
        # Debug info
        logger.debug(f"Input shape: {imgs.shape}")
        
        # Interpolate images for student if needed
        if imgs.shape[-1] != 224:
            student_imgs = torch.nn.functional.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
        else:
            student_imgs = imgs
            
        # Debug info
        logger.debug(f"Student input shape: {student_imgs.shape}")
            
        optimizer.zero_grad()
        
        s_logits = student(student_imgs)
            
        with torch.no_grad():
            t_logits = teacher(imgs)
            
        # Debug info
        logger.debug(f"Student logits shape: {s_logits.shape}")
        logger.debug(f"Teacher logits shape: {t_logits.shape}")
            
        loss = response_based_loss(s_logits, t_logits, labels)
            
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    return loss_sum / len(loader)

def evaluate(student, loader, device):
    student.eval()
    ids, labels, preds = [], [], []
    pbar = tqdm(loader, desc='Evaluating')
    
    with torch.no_grad():
        for imgs, lbls, img_ids in pbar:
            imgs = imgs.to(device)
            # Interpolate images for student if needed
            if imgs.shape[-1] != 224:
                imgs = torch.nn.functional.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
            logits = student(imgs)
            prob = torch.softmax(logits, dim=1)[:,1]
            
            # Debug logging
            logger.debug(f"Logits shape: {logits.shape}")
            logger.debug(f"Probabilities: {prob}")
            logger.debug(f"Labels: {lbls}")
            
            # Convert and check for NaN
            prob_np = prob.cpu().numpy()
            lbls_np = lbls.numpy()
            
            if np.any(np.isnan(prob_np)):
                logger.error(f"NaN found in predictions: {prob_np}")
            if np.any(np.isnan(lbls_np)):
                logger.error(f"NaN found in labels: {lbls_np}")
            
            preds.extend(prob_np)
            labels.extend(lbls_np)
            ids.extend(img_ids)
  
    # print(f"Ratio of positive class: ({np.sum(labels)}/{len(labels)})")
    # Create DataFrames with proper format
    sol = pd.DataFrame({'row_id': ids, 'target': labels})
    sub = pd.DataFrame({'row_id': ids, 'target': preds})
    
    # Check DataFrame types
    logger.debug(f"Solution DataFrame types: {sol.dtypes}")
    logger.debug(f"Submission DataFrame types: {sub.dtypes}")
    
    # Ensure labels are numeric
    sol['target'] = sol['target'].astype(float)
    sub['target'] = sub['target'].astype(float)
    
    # Final check for NaN
    if sol['target'].isna().any():
        logger.error("NaN found in solution targets")
    if sub['target'].isna().any():
        logger.error("NaN found in submission targets")
    
    return score(sol, sub, 'row_id')

def save_checkpoint(model, optimizer, epoch, val_metric, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metric': val_metric
    }
    
    torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pt')
    logger.info(f'Saved checkpoint for epoch {epoch}')

def plot_metrics(train_losses, val_metrics, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_metrics)
    plt.title('Validation pAUC')
    plt.xlabel('Epoch')
    plt.ylabel('pAUC')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_metrics.png')
    plt.close()

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Set parameters directly
    root_dir = '.'
    ckpt = 'last.ckpt'
    method = 'response'
    epochs = 20
    batch_size = 32
    learning_rate = 1e-4
    save_dir = 'checkpoints'
    debug = False

    # Set up logging
    if debug:
        logger.setLevel(logging.DEBUG)
    
    # Set up device and transforms
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets and dataloaders
    train_ds = SkinLesionDataset(root_dir, transform=transform, is_train=True)
    val_ds = SkinLesionDataset(root_dir, transform=transform, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load models
    teacher = load_teacher(ckpt, device)
    student = load_student(device)
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    # Training loop
    train_losses = []
    val_metrics = []
    best_val_metric = 0
    
    os.makedirs(save_dir, exist_ok=True)
    log_file_path = os.path.join(save_dir, 'training_log.csv')
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_pAUC'])

    for ep in range(1, epochs+1):
        logger.info(f'\nEpoch {ep}/{epochs}')
        
        # Train
        tr_loss = train_one_epoch(student, teacher, train_loader, optimizer, device, method)
        train_losses.append(tr_loss)
        
        # Evaluate
        val_metric = evaluate(student, val_loader, device)
        val_metrics.append(val_metric)
        
        # Update learning rate
        scheduler.step(val_metric)
        
        # Save checkpoint if best model
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            save_checkpoint(student, optimizer, ep, val_metric, save_dir)
        
        # Log metrics to file
        with open(log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep, f'{tr_loss:.4f}', f'{val_metric:.4f}'])
        
        logger.info(f'Epoch {ep}: Train Loss={tr_loss:.4f}, Val pAUC={val_metric:.4f}')
    
    # Plot and save metrics
    plot_metrics(train_losses, val_metrics, save_dir)
    logger.info('Training completed!')
