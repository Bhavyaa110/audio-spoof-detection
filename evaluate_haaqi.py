import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_curve
import numpy as np
from tqdm import tqdm

from models.haaqi_model import HAAQI_Spoof
from utils.asvspoof_loader import ASVspoofDataset
from config import *

# ===== DEVICE =====
print(f"Evaluating on: {DEVICE}")

# ===== DATA =====
eval_dataset = ASVspoofDataset(
    "asvspoof_dataset/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "asvspoof_dataset/ASVspoof2019_LA_dev/flac"
)

eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===== MODEL =====
model = HAAQI_Spoof().to(DEVICE)
model.load_state_dict(torch.load("outputs/haaqi_model.pth", map_location=DEVICE))
model.eval()

# ===== EER FUNCTION =====
def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return fpr[idx]

# ===== EVALUATION =====
all_preds, all_labels, all_scores = [], [], []

eval_bar = tqdm(eval_loader, desc="Evaluating", ncols=80)

with torch.no_grad():
    for x, y in eval_bar:
        x = x.to(DEVICE)

        # IMPORTANT shape fix
        if len(x.shape) == 3:
            x = x.squeeze(1)

        output = model(x).squeeze()

        scores = output.cpu().numpy()
        preds = (scores > 0.5).astype(int)

        all_scores.extend(scores)
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

# ===== METRICS =====
accuracy = accuracy_score(all_labels, all_preds) * 100
eer = compute_eer(all_labels, all_scores)

print("=" * 50)
print("HAAQI MODEL RESULTS")
print("=" * 50)
print(f"Total Samples : {len(all_labels)}")
print(f"Accuracy      : {accuracy:.2f}%")
print(f"EER           : {eer:.4f}")
print("=" * 50)

# ===== SAVE RESULTS =====
with open("outputs/results_haaqi.txt", "w") as f:
    f.write("HAAQI MODEL RESULTS\n")
    f.write("="*50 + "\n")
    f.write(f"Accuracy: {accuracy:.2f}%\n")
    f.write(f"EER: {eer:.4f}\n")

print("Results saved to outputs/results_haaqi.txt")