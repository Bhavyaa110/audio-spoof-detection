import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from models.deeprawnet import DeepRawNet
from utils.asvspoof_loader import ASVspoofDataset
from config import *

# ===== LOAD DATA =====
eval_dataset = ASVspoofDataset(
    "asvspoof_dataset/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "asvspoof_dataset/ASVspoof2019_LA_dev/flac"
)

eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

# ===== MODEL =====
model = DeepRawNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===== EVAL =====
preds = []
labels = []

with torch.no_grad():
    for x, y in eval_loader:
        x = x.to(DEVICE)

        output = model(x)
        pred = torch.argmax(output, dim=1)

        preds.append(pred.item())
        labels.append(y.item())

acc = accuracy_score(labels, preds)

print(f"\nAccuracy: {acc:.4f}")