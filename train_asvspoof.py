import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.deeprawnet import DeepRawNet
from utils.asvspoof_loader import ASVspoofDataset
from config import *

# ===== LOAD DATA =====
train_dataset = ASVspoofDataset(
    "asvspoof_dataset/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "asvspoof_dataset/ASVspoof2019_LA_train/flac"
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===== MODEL =====
model = DeepRawNet(dropout_rate=0.5).to(DEVICE)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("loading dataset")
# ===== TRAIN =====
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for i, (x, y) in enumerate(train_loader):
        print(f"Batch {i+1}")
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ===== SAVE =====
torch.save(model.state_dict(), MODEL_PATH)
print("Model saved!")