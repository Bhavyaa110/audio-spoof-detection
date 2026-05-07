import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_curve
import numpy as np
from tqdm import tqdm
from models.haaqi_model import HAAQI_Spoof
from utils.asvspoof_loader import ASVspoofDataset
from config import *
import os
import copy
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import logging
logging.set_verbosity_error()


# ==============================
# EER COMPUTATION
# ==============================
def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return fpr[idx]


# ==============================
# CLASS WEIGHT HELPER
# ==============================
def compute_pos_weight(dataset):
    """
    Computes pos_weight = num_negatives / num_positives
    for BCEWithLogitsLoss to handle class imbalance.
    Assumes dataset.labels is a list of 0/1 integers.
    """
    labels = np.array(dataset.labels)        # now works instantly
    num_pos = (labels == 0).sum()             # bonafide = 0
    num_neg = (labels == 1).sum()             # spoof    = 1
    pos_weight = num_neg / max(num_pos, 1)
    print(f"  Genuine (bonafide): {num_pos}  |  Spoof: {num_neg}  |  pos_weight: {pos_weight:.4f}")
    return torch.tensor([pos_weight], dtype=torch.float32)


# ==============================
# PARTIAL UNFREEZE
# ==============================
def setup_model_freezing(model, unfreeze_last_n=4):
    """
    Freeze all wav2vec layers, then unfreeze the last N transformer blocks.
    This allows fine-tuning the top layers while keeping lower layers stable.
    """
    # Freeze everything first
    for param in model.wav2vec.parameters():
        param.requires_grad = False

    # Unfreeze the last N encoder transformer layers
    encoder_layers = model.wav2vec.encoder.layers
    num_layers = len(encoder_layers)
    unfreeze_from = max(0, num_layers - unfreeze_last_n)

    for layer in encoder_layers[unfreeze_from:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Count trainable params
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  Unfrozen wav2vec layers: last {unfreeze_last_n} of {num_layers}")


# ==============================
# OPTIMIZER WITH LAYER-WISE LR
# ==============================
def build_optimizer(model, backbone_lr=1e-5, head_lr=1e-3):
    """
    Use a smaller LR for the partially unfrozen wav2vec layers
    and a larger LR for the classification head.
    """
    backbone_params = [
        p for n, p in model.named_parameters()
        if "wav2vec" in n and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if "wav2vec" not in n and p.requires_grad
    ]
    return torch.optim.Adam([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ])


# ==============================
# MAIN
# ==============================
if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True

    # ==============================
    # DEVICE INFO
    # ==============================
    print(f"Training on : {DEVICE}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name    : {torch.cuda.get_device_name(0)}")
    print("-" * 50)

    # ==============================
    # DATA
    # ==============================
    train_dataset = ASVspoofDataset(
        "asvspoof_dataset/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        "asvspoof_dataset/ASVspoof2019_LA_train/flac"
    )

    val_dataset = ASVspoofDataset(
        "asvspoof_dataset/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        "asvspoof_dataset/ASVspoof2019_LA_dev/flac"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Training samples  : {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("-" * 50)

    # ==============================
    # CLASS IMBALANCE
    # ==============================
    print("Computing class weights...")
    pos_weight = compute_pos_weight(train_dataset).to(DEVICE)
    print("-" * 50)

    # ==============================
    # MODEL
    # ==============================
    model = HAAQI_Spoof().to(DEVICE)

    print("Setting up model freezing...")
    setup_model_freezing(model, unfreeze_last_n=4)
    print("-" * 50)

    # ==============================
    # LOSS, OPTIMIZER, SCHEDULER
    # ==============================
    # BCEWithLogitsLoss = sigmoid + BCE internally — more numerically stable
    # pos_weight handles class imbalance
    # NOTE: Remove sigmoid from model's final layer if you had one,
    #       since BCEWithLogitsLoss applies it internally.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = build_optimizer(model, backbone_lr=1e-5, head_lr=1e-3)

    # CosineAnnealingLR smoothly reduces LR from initial to eta_min
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # ==============================
    # EARLY STOPPING SETUP
    # ==============================
    PATIENCE         = 5          # increased from 5 — gives model more time
    best_val_eer     = float("inf")  # track EER instead of loss (primary metric)
    best_val_loss    = float("inf")
    patience_counter = 0
    best_model_state = None

    # ==============================
    # HISTORY
    # ==============================
    history = {
        "train_loss"    : [],
        "train_accuracy": [],
        "train_eer"     : [],
        "val_loss"      : [],
        "val_accuracy"  : [],
        "val_eer"       : [],
    }

    print("Starting Training...\n")

    # ==============================
    # TRAIN LOOP
    # ==============================
    for epoch in range(EPOCHS):

        # ===== TRAIN =====
        model.train()
        total_loss = 0
        train_preds, train_labels, train_scores = [], [], []

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", ncols=100)

        for x, y in train_bar:
            x = x.to(DEVICE)
            y = y.to(DEVICE).float()

            if len(x.shape) == 3:
                x = x.squeeze(1)

            # Model output = raw logits (no sigmoid in model now)
            output = model(x).squeeze()
            loss   = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping — prevents exploding gradients during fine-tuning
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            # Apply sigmoid manually for metrics since model outputs logits
            scores = torch.sigmoid(output).detach().cpu().numpy()
            preds  = (scores > 0.5).astype(int)

            train_scores.extend(scores)
            train_preds.extend(preds)
            train_labels.extend(y.cpu().numpy())

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_acc      = accuracy_score(train_labels, train_preds)
        train_eer      = compute_eer(train_labels, train_scores)

        # Step the LR scheduler after each epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()

        # ===== VALIDATION =====
        model.eval()
        val_loss   = 0
        val_preds, val_labels, val_scores = [], [], []

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]  ", ncols=100)

        with torch.no_grad():
            for x, y in val_bar:
                x = x.to(DEVICE)
                y = y.to(DEVICE).float()

                if len(x.shape) == 3:
                    x = x.squeeze(1)

                output   = model(x).squeeze()
                loss     = criterion(output, y)
                val_loss += loss.item()

                scores = torch.sigmoid(output).cpu().numpy()
                preds  = (scores > 0.5).astype(int)

                val_scores.extend(scores)
                val_preds.extend(preds)
                val_labels.extend(y.cpu().numpy())

                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)
        val_acc      = accuracy_score(val_labels, val_preds)
        val_eer      = compute_eer(val_labels, val_scores)

        # ===== STORE HISTORY =====
        history["train_loss"].append(avg_train_loss)
        history["train_accuracy"].append(train_acc * 100)
        history["train_eer"].append(train_eer)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_acc * 100)
        history["val_eer"].append(val_eer)

        # ===== PRINT =====
        print("\n" + "=" * 50)
        print(f"Epoch {epoch+1}  |  LR: {current_lr}")
        print(f"  Train Loss : {avg_train_loss:.4f}  |  Train Acc: {train_acc*100:.2f}%  |  Train EER: {train_eer:.4f}")
        print(f"  Val Loss   : {avg_val_loss:.4f}  |  Val Acc  : {val_acc*100:.2f}%  |  Val EER  : {val_eer:.4f}")
        print("=" * 50)

        # ===== EARLY STOPPING (on EER — primary metric for spoof detection) =====
        if val_eer < best_val_eer:
            best_val_eer     = val_eer
            best_val_loss    = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  [Checkpoint] Val EER improved to {best_val_eer:.4f} — saving best model.")
        else:
            patience_counter += 1
            print(f"  [Early Stop] No EER improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping triggered at epoch {epoch+1}!")
                break

        print()

    # ==============================
    # FINAL SUMMARY
    # ==============================
    print("=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"  Best Val EER      : {best_val_eer:.4f}")
    print(f"  Best Val Loss     : {best_val_loss:.4f}")
    print(f"  Final Train Acc   : {history['train_accuracy'][-1]:.2f}%")
    print(f"  Final Train EER   : {history['train_eer'][-1]:.4f}")
    print(f"  Final Val Acc     : {history['val_accuracy'][-1]:.2f}%")
    print(f"  Final Val EER     : {history['val_eer'][-1]:.4f}")
    print("=" * 50)

    # ==============================
    # SAVE MODEL & RESULTS
    # ==============================
    os.makedirs("outputs", exist_ok=True)

    torch.save(best_model_state, "outputs/haaqi_model.pth")
    print("\nBest model saved to outputs/haaqi_model.pth")

    # Save results to txt
    with open("outputs/haaqi_results.txt", "w") as f:
        f.write("TRAINING HISTORY\n")
        f.write("=" * 50 + "\n")
        for ep in range(len(history["train_loss"])):
            f.write(f"Epoch {ep+1}\n")
            f.write(f"  Train Loss : {history['train_loss'][ep]:.4f}\n")
            f.write(f"  Train Acc  : {history['train_accuracy'][ep]:.2f}%\n")
            f.write(f"  Train EER  : {history['train_eer'][ep]:.4f}\n")
            f.write(f"  Val Loss   : {history['val_loss'][ep]:.4f}\n")
            f.write(f"  Val Acc    : {history['val_accuracy'][ep]:.2f}%\n")
            f.write(f"  Val EER    : {history['val_eer'][ep]:.4f}\n")
            f.write("-" * 50 + "\n")
        f.write("\nFINAL RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"  Best Val EER    : {best_val_eer:.4f}\n")
        f.write(f"  Best Val Loss   : {best_val_loss:.4f}\n")
        f.write(f"  Final Train Acc : {history['train_accuracy'][-1]:.2f}%\n")
        f.write(f"  Final Train EER : {history['train_eer'][-1]:.4f}\n")
        f.write(f"  Final Val Acc   : {history['val_accuracy'][-1]:.2f}%\n")
        f.write(f"  Final Val EER   : {history['val_eer'][-1]:.4f}\n")

    print("Results saved to outputs/haaqi_results.txt")