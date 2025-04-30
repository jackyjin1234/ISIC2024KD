import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

# Distillation components
from distill_reponse import (
    SkinLesionDataset,
    load_teacher,
    load_student,
    response_based_loss,
    evaluate,
    save_checkpoint,
    plot_metrics,
)  # :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}

from feature_matching import feature_matching_loss    # implement in feature_matching.ipynb
from TokenKD import token_based_loss                  # implement in TokenKD.ipynb

# GBDT downstream
from GBDT import extract_features, merge_meta, train_gbdt  # :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}

def train_one_epoch_combined(student, teacher, loader, optimizer, device,
                             w_feat, w_token, w_resp, T_temp, alpha_resp):
    student.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Train combined KD")
    for imgs, labels, _ in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # Student forward
        s_logits = student(imgs)
        # Teacher forward (no grad)
        with torch.no_grad():
            t_logits = teacher(imgs)

        # 1) Response‐based KD
        loss_resp = response_based_loss(s_logits, t_logits, labels,
                                        T=T_temp, alpha=alpha_resp)

        # 2) Feature‐matching KD
        # Assumes feature_matching_loss returns a scalar given (student, teacher, imgs)
        loss_feat = feature_matching_loss(student, teacher, imgs, labels)

        # 3) Token‐based KD
        # Assumes token_based_loss returns a scalar given (student, teacher, imgs, labels)
        loss_token = token_based_loss(student, teacher, imgs, labels)

        # Weighted sum
        loss = w_feat * loss_feat + w_token * loss_token + w_resp * loss_resp
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "resp": f"{loss_resp.item():.4f}",
            "feat": f"{loss_feat.item():.4f}",
            "token": f"{loss_token.item():.4f}",
        })

    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(
        description="Combined KD + GBDT pipeline for skin lesion classification"
    )
    parser.add_argument("--root_dir", type=str, default=".",
                        help="root directory for images") 
    parser.add_argument("--meta_csv", type=str, required=True,
                        help="path to train-metadata.csv")
    parser.add_argument("--h5_file", type=str, required=True,
                        help="path to train-image.hdf5")
    parser.add_argument("--teacher_ckpt", type=str, required=True,
                        help="checkpoint for pretrained teacher")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--w_feat", type=float, default=1.0,
                        help="weight for feature‐matching loss")
    parser.add_argument("--w_token", type=float, default=1.0,
                        help="weight for token‐based loss")
    parser.add_argument("--w_resp", type=float, default=0.6,
                        help="weight for response‐based loss")
    parser.add_argument("--T_temp", type=float, default=4.0,
                        help="temperature for response‐based KD")
    parser.add_argument("--alpha_resp", type=float, default=0.6,
                        help="alpha for response‐based KD mix")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Transforms
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # Data
    train_ds = SkinLesionDataset(args.root_dir, transform=transform, is_train=True)
    val_ds   = SkinLesionDataset(args.root_dir, transform=transform, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Models
    teacher = load_teacher(args.teacher_ckpt, device)
    student = load_student(device)

    # Optimizer & scheduler
    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                     patience=2, factor=0.5)

    # Training loop
    best_val = 0.0
    train_losses, val_metrics = [], []

    # CSV log
    log_csv = Path(args.save_dir) / "train_log.csv"
    with open(log_csv, "w") as f:
        f.write("epoch,train_loss,val_pAUC\n")

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        tr_loss = train_one_epoch_combined(
            student, teacher, train_loader, optimizer, device,
            args.w_feat, args.w_token, args.w_resp,
            args.T_temp, args.alpha_resp
        )
        train_losses.append(tr_loss)

        val_pauc = evaluate(student, val_loader, device)
        val_metrics.append(val_pauc)

        scheduler.step(val_pauc)

        # Save best
        if val_pauc > best_val:
            best_val = val_pauc
            save_checkpoint(student, optimizer, epoch, val_pauc, args.save_dir)

        # Log
        with open(log_csv, "a") as f:
            f.write(f"{epoch},{tr_loss:.4f},{val_pauc:.4f}\n")

    # Plot metrics
    plot_metrics(train_losses, val_metrics, args.save_dir)

    feats, labels, ids = extract_features(
        student, args.meta_csv, args.h5_file, out_npy="features.npy"
    )

    # 2) merge with metadata
    X, y = merge_meta(args.meta_csv, feats, ids)

    # 3) train GBDT
    clf = train_gbdt(X, y)
    clf.save_model("xgb_gbdt_final.json")
    print("[INFO] Combined pipeline completed successfully.")

if __name__ == "__main__":
    main()
