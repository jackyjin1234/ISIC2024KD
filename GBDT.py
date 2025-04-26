#!/usr/bin/env python3

import os
import io
import math
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# 1. Define the Student architecture (matches notebook, returns intermediate features)
class DistillableViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=2, dim=384):
        super().__init__()
        self.vit = timm.create_model(
            'vit_small_patch16_224',
            pretrained=False,
            num_classes=num_classes,
            img_size=image_size,
            patch_size=patch_size
        )
        self.dim = dim

    def forward(self, x):
        B = x.shape[0]
        x = self.vit.patch_embed(x)                  # (B, N, dim)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)          # (B, 1+N, dim)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        mid_feats = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in (3, 7):
                # mean over all patch tokens
                mid_feats.append(x[:, 1:, :].mean(dim=1))

        x = self.vit.norm(x)
        # final layer patch features
        final_feat = x[:, 1:, :].mean(dim=1)
        mid_feats.append(final_feat)

        logits = self.vit.head(x[:, 0])
        return logits, mid_feats


# -----------------------------------------------------------------------------
# 2. Dataset: read image bytes from HDF5 by isic_id and convert to PIL Image
class SkinDataset(Dataset):
    def __init__(self, csv_file, hdf5_file, transform=None):
        self.df = pd.read_csv(csv_file, low_memory=False)
        self.h5 = h5py.File(hdf5_file, 'r')
        self.transform = transform

        self.ids = self.df['isic_id'].astype(str).values
        if 'target' in self.df.columns:
            self.targets = self.df['target'].astype(int).values
        else:
            self.targets = np.zeros(len(self.ids), dtype=int)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        isic_id = self.ids[idx]
        label = self.targets[idx]
        img_bytes = self.h5[isic_id][()]
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, isic_id


# -----------------------------------------------------------------------------
# 3. Load Student model, adjust pos_embed if needed, and load checkpoint
def load_student_and_checkpoint(weight_path):
    model = DistillableViT(image_size=224).to(DEVICE)

    # 1. Load checkpoint and extract state_dict
    ckpt = torch.load(weight_path, map_location='cpu')
    raw = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

    # 2. Remove 'module.' prefix from keys
    state = {k.replace('module.', ''): v for k, v in raw.items()}

    # 3. If keys are not prefixed with 'vit.', add it
    if not any(k.startswith('vit.') for k in state):
        state = {f'vit.{k}': v for k, v in state.items()}

    # 4. Find the pos_embed key and interpolate if dimensions mismatch
    pe_keys = [k for k in state if 'pos_embed' in k]
    if pe_keys:
        pe_key = pe_keys[0]
        pe_model = model.vit.pos_embed.shape[1]
        pe_ckpt = state[pe_key].shape[1]

        if pe_model != pe_ckpt:
            print(f"⚠️  Interpolating {pe_key}: model expects {pe_model}, checkpoint has {pe_ckpt}")

            cls_tok = state[pe_key][:, :1]
            grid_old = state[pe_key][:, 1:]
            old_n = int(math.sqrt(grid_old.shape[1]))
            new_n = int(math.sqrt(pe_model - 1))

            grid_old = grid_old.reshape(1, old_n, old_n, -1).permute(0, 3, 1, 2)
            grid_new = torch.nn.functional.interpolate(
                grid_old, size=(new_n, new_n), mode='bicubic', align_corners=False
            )
            grid_new = grid_new.permute(0, 2, 3, 1).reshape(1, new_n * new_n, -1)
            state[pe_key] = torch.cat([cls_tok, grid_new], dim=1)
    else:
        print("⚠️  No pos_embed key found in checkpoint; skipping interpolation.")

    # 5. Strictly load the state_dict
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"After interpolation got missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


# -----------------------------------------------------------------------------
# 4. Extract and cache features
def extract_features(model, csv_file, hdf5_file, out_npy='feat.npy', batch_size=64):
    if Path(out_npy).exists():
        data = np.load(out_npy, allow_pickle=True).item()
        return data['feats'], data['labels'], data['ids']

    tf = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    ds = SkinDataset(csv_file, hdf5_file, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    all_feats, all_labels, all_ids = [], [], []
    with torch.no_grad():
        for imgs, labels, ids in tqdm(loader, desc='ExtractFeatures'):
            imgs = imgs.to(DEVICE)
            logits, mid_feats = model(imgs)
            # Concatenate the three feature vectors
            feats = torch.cat(mid_feats, dim=1).cpu().numpy()
            all_feats.append(feats)
            all_labels.extend(labels)
            all_ids.extend(ids)

    feats = np.vstack(all_feats)
    np.save(out_npy, {'feats': feats, 'labels': np.array(all_labels), 'ids': np.array(all_ids)})
    return feats, np.array(all_labels), np.array(all_ids)


# -----------------------------------------------------------------------------
# 5. Merge metadata with features
def merge_meta(meta_csv, feats, ids):
    df = pd.read_csv(meta_csv, low_memory=False).set_index('isic_id')
    df = df.loc[ids]
    y = df['target'].values
    df = df.drop(columns=['target'])

    # One-hot encode categorical columns and standardize numerical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=cat_cols)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])

    X = np.concatenate([feats, df.values], axis=1)
    return X, y


# -----------------------------------------------------------------------------
# 6. Train and validate GBDT
def train_gbdt(X, y):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scale_pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    clf = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        tree_method='gpu_hist' if DEVICE.type=='cuda' else 'hist'
    )
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    pred = clf.predict_proba(X_val)[:, 1]
    print(f'[RESULT] Validation ROC‑AUC = {roc_auc_score(y_val, pred):.4f}')
    return clf


# -----------------------------------------------------------------------------
# 7. Main routine
if __name__ == '__main__':
    CSV_FILE = '/kaggle/input/isic-2024-challenge/train-metadata.csv'
    H5_FILE = '/kaggle/input/isic-2024-challenge/train-image.hdf5'
    WEIGHT = '/kaggle/input/studentmodel/best_model.pth'

    # 7.1 Load student model and checkpoint
    student = load_student_and_checkpoint(WEIGHT)

    # 7.2 Extract features
    feats, labels, ids = extract_features(student, CSV_FILE, H5_FILE, out_npy='features.npy')

    # 7.3 Merge metadata
    X, y = merge_meta(CSV_FILE, feats, ids)

    # 7.4 Train GBDT
    gbdt = train_gbdt(X, y)

    # 7.5 Save the trained GBDT model
    gbdt.save_model('xgb_gbdt_final.json')
    print('✅ GBDT model saved to xgb_gbdt_final.json')
