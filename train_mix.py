import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_curve
from torch.nn.utils.rnn import pad_sequence
from models.mix_deeprawnet import MixDeepRawNet
from utils.mix_loader import MixDataset
from config import *

print(f"Using Device : {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU : {torch.cuda.get_device_name(0)}"
    )
else:
    print("Running on CPU")
def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2 * 100
    return eer



train_dataset = MixDataset(
    "train"
)

val_dataset = MixDataset(
    "val"
)
def collate_fn(batch):

    feats, labels = zip(*batch)

    feats = pad_sequence(
        feats,
        batch_first=True
    )

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    return feats, labels

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)
print(
    f"Train Samples : {len(train_dataset)}"
)
print(
    f"Val Samples   : {len(val_dataset)}"
)

model = MixDeepRawNet().to(DEVICE)

criterion = nn.NLLLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

best_val_eer = 999

history = {
    "train_acc":[],
    "train_eer":[],
    "val_acc":[],
    "val_eer":[]
}

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # TRAIN
    model.train()

    train_correct = 0
    train_total = 0
    train_scores = []
    train_labels = []

    train_bar = tqdm(train_loader)

    for x,y in train_bar:

        x=x.to(DEVICE)
        y=y.to(DEVICE)

        optimizer.zero_grad()

        out=model(x)

        loss=criterion(out,y)

        loss.backward()
        optimizer.step()

        prob=torch.exp(out)
        pred=out.argmax(1)

        train_correct += (pred==y).sum().item()
        train_total += y.size(0)

        train_scores.extend(
            prob[:,1].detach().cpu().tolist()
        )

        train_labels.extend(
            y.cpu().tolist()
        )

    train_acc = 100 * train_correct/train_total
    train_eer = compute_eer(
        train_labels,
        train_scores
    )

    # VALIDATION
    model.eval()

    val_correct=0
    val_total=0
    val_scores=[]
    val_labels=[]

    with torch.no_grad():

        for x,y in val_loader:

            x=x.to(DEVICE)
            y=y.to(DEVICE)

            out=model(x)

            prob=torch.exp(out)
            pred=out.argmax(1)

            val_correct += (pred==y).sum().item()
            val_total += y.size(0)

            val_scores.extend(
                prob[:,1].cpu().tolist()
            )

            val_labels.extend(
                y.cpu().tolist()
            )

    val_acc = 100 * val_correct/val_total

    val_eer = compute_eer(
        val_labels,
        val_scores
    )

    history["train_acc"].append(train_acc)
    history["train_eer"].append(train_eer)
    history["val_acc"].append(val_acc)
    history["val_eer"].append(val_eer)

    print("="*50)
    print(f"Train Accuracy : {train_acc:.2f}%")
    print(f"Train EER      : {train_eer:.2f}%")
    print(f"Val Accuracy   : {val_acc:.2f}%")
    print(f"Val EER        : {val_eer:.2f}%")
    print("="*50)

    if val_eer < best_val_eer:
        best_val_eer = val_eer
        torch.save(
            model.state_dict(),
            "outputs/mix_model.pth"
        )
        print("Best model saved")

os.makedirs(
    "outputs",
    exist_ok=True
)

with open(
    "outputs/mix_results.txt",
    "w"
) as f:

    f.write("FINAL RESULTS\n")
    f.write("="*50+"\n")
    f.write(
        f"Training Accuracy : {history['train_acc'][-1]:.2f}%\n"
    )
    f.write(
        f"Training EER : {history['train_eer'][-1]:.2f}%\n"
    )
    f.write(
        f"Validation Accuracy : {history['val_acc'][-1]:.2f}%\n"
    )
    f.write(
        f"Validation EER : {history['val_eer'][-1]:.2f}%\n"
    )

print("\nResults saved")