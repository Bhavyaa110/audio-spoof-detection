import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from models.deeprawnet import DeepRawNet
from utils.asvspoof_loader import ASVspoofDataset
from config import *

# ===== LOAD DATA =====
eval_dataset = ASVspoofDataset(
    "asvspoof_dataset/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "asvspoof_dataset/ASVspoof2019_LA_dev/flac"
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
preds  = []
labels = []

eval_bar = tqdm(eval_loader, desc="  Evaluating", unit="batch", ncols=80)

with torch.no_grad():
    for x, y in eval_bar:
        x = x.to(DEVICE)

        output = model(x)
        pred   = torch.argmax(output, dim=1)

        preds.extend(pred.cpu().tolist())
        labels.extend(y.tolist())

accuracy   = accuracy_score(labels, preds) * 100
error_rate = 100 - accuracy

print("=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"  Total Samples : {len(labels)}")
print(f"  Accuracy      : {accuracy:.2f}%")
print(f"  Error Rate    : {error_rate:.2f}%")
print("=" * 50)