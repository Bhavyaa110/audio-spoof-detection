import torch
import numpy as np
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from models.mix_deeprawnet import MixDeepRawNet
from utils.mix_loader import MixDataset
from config import *

def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr-fnr))
    eer=(fpr[idx]+fnr[idx])/2*100
    return eer

dataset = MixDataset(
    "test"
)

loader = DataLoader(
    dataset,
    batch_size=2
)

model = MixDeepRawNet().to(
    DEVICE
)

model.load_state_dict(
    torch.load(
        "outputs/mix_model.pth",
        map_location=DEVICE
    )
)

model.eval()

correct=0
total=0

scores=[]
labels=[]

with torch.no_grad():

    for x,y in loader:

        x=x.to(DEVICE)
        y=y.to(DEVICE)

        out=model(x)

        prob=torch.exp(out)
        pred=out.argmax(1)

        correct += (pred==y).sum().item()
        total += y.size(0)

        scores.extend(
            prob[:,1].cpu().tolist()
        )

        labels.extend(
            y.cpu().tolist()
        )

acc = 100*correct/total

eer = compute_eer(
    labels,
    scores
)

print("="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"Evaluation Accuracy : {acc:.2f}%")
print(f"Evaluation EER      : {eer:.2f}%")
print("="*50)

with open(
    "outputs/mix_eval.txt",
    "w"
) as f:

    f.write(
        f"Evaluation Accuracy : {acc:.2f}%\n"
    )
    f.write(
        f"Evaluation EER : {eer:.2f}%\n"
    )