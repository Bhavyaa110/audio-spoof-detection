import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import accuracy_score, roc_curve
import numpy as np
from tqdm import tqdm
from models.haaqi_model import HAAQI_Spoof
from utils.asvspoof_loader import ASVspoofDataset
from config import *
import os
import warnings

warnings.filterwarnings("ignore")

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import logging
logging.set_verbosity_error()
# ==============================
# SPEED (CPU safe)
# ==============================
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
# DATA (FIXED 10K → 8K/2K)
# ==============================
full_dataset = ASVspoofDataset(
    "asvspoof_dataset/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "asvspoof_dataset/ASVspoof2019_LA_train/flac"
)

MAX_SAMPLES = 10000
indices = list(range(len(full_dataset)))

np.random.seed(42)
np.random.shuffle(indices)

subset_indices = indices[:MAX_SAMPLES]
full_dataset = Subset(full_dataset, subset_indices)

train_size = 8000
val_size = 2000

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# ==============================
# DATALOADER (Windows safe)
# ==============================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

print(f"Total samples     : {len(full_dataset)}")
print(f"Training samples  : {train_size}")
print(f"Validation samples: {val_size}")
print("-" * 50)

# ==============================
# MODEL
# ==============================
model = HAAQI_Spoof().to(DEVICE)

# Freeze wav2vec (huge speed boost)
for param in model.wav2vec.parameters():
    param.requires_grad = False

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ==============================
# EER FUNCTION
# ==============================
def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return fpr[idx]

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

        output = model(x).squeeze()
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        scores = output.detach().cpu().numpy()
        preds = (scores > 0.5).astype(int)

        train_scores.extend(scores)
        train_preds.extend(preds)
        train_labels.extend(y.cpu().numpy())

        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    train_acc = accuracy_score(train_labels, train_preds)
    train_eer = compute_eer(train_labels, train_scores)

    # ===== VALIDATION =====
    model.eval()
    val_preds, val_labels, val_scores = [], [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).float()

            if len(x.shape) == 3:
                x = x.squeeze(1)

            output = model(x).squeeze()

            scores = output.cpu().numpy()
            preds = (scores > 0.5).astype(int)

            val_scores.extend(scores)
            val_preds.extend(preds)
            val_labels.extend(y.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_eer = compute_eer(val_labels, val_scores)

    # ===== PRINT =====
    print("\n" + "=" * 50)
    print(f"Epoch {epoch+1}")
    print(f"Train Loss : {total_loss:.4f}")
    print(f"Train Acc  : {train_acc*100:.2f}%")
    print(f"Train EER  : {train_eer:.4f}")
    print(f"Val Acc    : {val_acc*100:.2f}%")
    print(f"Val EER    : {val_eer:.4f}")
    print("=" * 50)

# ==============================
# SAVE MODEL
# ==============================
torch.save(model.state_dict(), "outputs/haaqi_model.pth")
print("\nModel saved to outputs/haaqi_model.pth")