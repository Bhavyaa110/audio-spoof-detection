import torch
import torch.nn as nn

class MixDeepRawNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.gru = nn.GRU(
            input_size=1536,
            hidden_size=512,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(
            512,
            256
        )

        self.fc2 = nn.Linear(
            256,
            2
        )

        self.relu = nn.ReLU()

        self.logsoftmax = nn.LogSoftmax(
            dim=1
        )

    def forward(self,x):

        out,_ = self.gru(x)

        out = out[:,-1,:]

        out = self.dropout(out)

        out = self.relu(
            self.fc1(out)
        )

        out = self.fc2(out)

        return self.logsoftmax(out)