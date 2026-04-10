import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.deeprawnet import DeepRawNet
from utils.asvspoof_loader import ASVspoofDataset
from config import *

# ===== DEVICE INFO =====
print(f"Training on : {DEVICE}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name    : {torch.cuda.get_device_name(0)}")
print("-" * 50)

# ===== LOAD DATA (10000 samples, 80/20 split) =====
full_dataset = ASVspoofDataset(
    "asvspoof_dataset/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "asvspoof_dataset/ASVspoof2019_LA_train/flac"
)

train_size = int(0.8 * len(full_dataset))   # 8000 samples
val_size   = len(full_dataset) - train_size  # 2000 samples
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

print(f"Total samples    : {len(full_dataset)}")
print(f"Training samples : {train_size}")
print(f"Validation samples: {val_size}")
print("-" * 50)

# ===== MODEL =====
model     = DeepRawNet(dropout_rate=0.3).to(DEVICE)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ===== EARLY STOPPING SETUP =====
PATIENCE        = 5       # stop if val loss does not improve for 5 epochs
best_val_loss   = float("inf")
patience_counter = 0
best_model_state = None

# ===== RESULTS STORAGE =====
history = {
    "train_loss"    : [],
    "train_accuracy": [],
    "train_error"   : [],
    "val_accuracy"  : [],
    "val_error"     : [],
    "val_loss"      : [],
}

print("Starting Training...\n")

# ===== TRAINING LOOP =====
for epoch in range(EPOCHS):
    print(f"Epoch [{epoch+1}/{EPOCHS}]")

    # ---- Training ----
    model.train()
    total_loss   = 0
    train_correct = 0
    train_total   = 0

    train_bar = tqdm(train_loader, desc="  Training  ", unit="batch", ncols=80)

    for x, y in train_bar:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        output = model(x)
        loss   = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predicted     = torch.argmax(output, dim=1)
        train_correct += (predicted == y).sum().item()
        train_total   += y.size(0)

        # Update loading bar with live loss
        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = (train_correct / train_total) * 100
    train_error    = 100 - train_accuracy

    # ---- Validation ----
    model.eval()
    val_loss     = 0
    val_correct  = 0
    val_total    = 0

    val_bar = tqdm(val_loader, desc="  Validating", unit="batch", ncols=80)

    with torch.no_grad():
        for x, y in val_bar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            output   = model(x)
            loss     = criterion(output, y)
            val_loss += loss.item()

            predicted   = torch.argmax(output, dim=1)
            val_correct += (predicted == y).sum().item()
            val_total   += y.size(0)

            val_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_val_loss  = val_loss / len(val_loader)
    val_accuracy  = (val_correct / val_total) * 100
    val_error     = 100 - val_accuracy

    # ---- Store history ----
    history["train_loss"].append(avg_train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["train_error"].append(train_error)
    history["val_loss"].append(avg_val_loss)
    history["val_accuracy"].append(val_accuracy)
    history["val_error"].append(val_error)

    # ---- Print epoch summary ----
    print(f"  Train Loss     : {avg_train_loss:.4f}")
    print(f"  Train Accuracy : {train_accuracy:.2f}%")
    print(f"  Train Error    : {train_error:.2f}%")
    print(f"  Val Loss       : {avg_val_loss:.4f}")
    print(f"  Val Accuracy   : {val_accuracy:.2f}%")
    print(f"  Val Error      : {val_error:.2f}%")
    print("-" * 50)

    # ---- Early Stopping Check ----
    if avg_val_loss < best_val_loss:
        best_val_loss    = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        print(f"  [Early Stop] Val loss improved to {best_val_loss:.4f} — saving best model.")
    else:
        patience_counter += 1
        print(f"  [Early Stop] No improvement. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch+1}!")
            break

    print()

# ===== FINAL SUMMARY =====
print("=" * 50)
print("FINAL RESULTS")
print("=" * 50)
print(f"  Best Val Loss     : {best_val_loss:.4f}")
print(f"  Final Train Accuracy : {history['train_accuracy'][-1]:.2f}%")
print(f"  Final Train Error    : {history['train_error'][-1]:.2f}%")
print(f"  Final Val Accuracy   : {history['val_accuracy'][-1]:.2f}%")
print(f"  Final Val Error      : {history['val_error'][-1]:.2f}%")
print("=" * 50)

# ===== SAVE MODEL AS .pth =====
os.makedirs("outputs", exist_ok=True)

# Save best model weights as .pth
# ===== SAVE =====
torch.save(best_model_state, MODEL_PATH)
print("\nBest model saved to outputs/deeprawnet.pth")

# Save training history to results.txt
with open("outputs/results.txt", "w") as f:
    f.write("TRAINING HISTORY\n")
    f.write("=" * 50 + "\n")
    for epoch in range(len(history["train_loss"])):
        f.write(f"Epoch {epoch+1}\n")
        f.write(f"  Train Loss     : {history['train_loss'][epoch]:.4f}\n")
        f.write(f"  Train Accuracy : {history['train_accuracy'][epoch]:.2f}%\n")
        f.write(f"  Train Error    : {history['train_error'][epoch]:.2f}%\n")
        f.write(f"  Val Loss       : {history['val_loss'][epoch]:.4f}\n")
        f.write(f"  Val Accuracy   : {history['val_accuracy'][epoch]:.2f}%\n")
        f.write(f"  Val Error      : {history['val_error'][epoch]:.2f}%\n")
        f.write("-" * 50 + "\n")
    f.write("\nFINAL RESULTS\n")
    f.write("=" * 50 + "\n")
    f.write(f"  Best Val Loss        : {best_val_loss:.4f}\n")
    f.write(f"  Final Train Accuracy : {history['train_accuracy'][-1]:.2f}%\n")
    f.write(f"  Final Train Error    : {history['train_error'][-1]:.2f}%\n")
    f.write(f"  Final Val Accuracy   : {history['val_accuracy'][-1]:.2f}%\n")
    f.write(f"  Final Val Error      : {history['val_error'][-1]:.2f}%\n")

print("Results saved to outputs/results.txt")