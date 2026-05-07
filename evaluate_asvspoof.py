import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

from models.deeprawnet import DeepRawNet
from utils.asvspoof_loader import ASVspoofDataset
from config import *

# ===== EER CALCULATION =====
def compute_eer(labels, scores):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2 * 100
    return eer

# ===== LOAD DATA =====
eval_dataset = ASVspoofDataset(
    "asvspoof_dataset/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
    "asvspoof_dataset/ASVspoof2019_LA_eval/flac"
)

eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Evaluation samples : {len(eval_dataset)}")
print(f"Running on         : {DEVICE}")
print("-" * 50)

# ===== MODEL =====
model = DeepRawNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===== EVAL =====
preds   = []
labels  = []
scores  = []   # probability scores needed for EER

eval_bar = tqdm(eval_loader, desc="  Evaluating", unit="batch", ncols=80)

with torch.no_grad():
    for x, y in eval_bar:
        x = x.to(DEVICE)

        output = model(x)                          # log probabilities
        prob   = torch.exp(output)                 # convert to probabilities
        pred   = torch.argmax(output, dim=1)

        preds.extend(pred.cpu().tolist())
        labels.extend(y.tolist())
        scores.extend(prob[:, 1].cpu().tolist())   # spoof probability score

accuracy   = accuracy_score(labels, preds) * 100
error_rate = 100 - accuracy
eer        = compute_eer(labels, scores)

print("=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"  Total Samples : {len(labels)}")
print(f"  Accuracy      : {accuracy:.2f}%")
print(f"  Error Rate    : {error_rate:.2f}%")
print(f"  EER           : {eer:.2f}%")
print("=" * 50)

os.makedirs("outputs", exist_ok=True)
with open("outputs/eval_results.txt", "w") as f:
    f.write("EVALUATION RESULTS\n")
    f.write("=" * 50 + "\n")
    f.write(f"  Total Samples : {len(labels)}\n")
    f.write(f"  Accuracy      : {accuracy:.2f}%\n")
    f.write(f"  Error Rate    : {error_rate:.2f}%\n")
    f.write(f"  EER           : {eer:.2f}%\n")
    f.write("=" * 50 + "\n")

print("Results saved to outputs/eval_results.txt")