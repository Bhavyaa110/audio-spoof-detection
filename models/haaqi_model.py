import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class BLSTM(nn.Module):
    def __init__(self, input_size=768, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out

class Attention(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        context = torch.sum(weights * x, dim=1)
        return context

class HAAQI_Spoof(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.blstm = BLSTM()
        self.attn = Attention()
        self.fc = nn.Linear(256, 1)

    def forward(self, waveform):
        with torch.no_grad():
            features = self.wav2vec(waveform).last_hidden_state

        x = self.blstm(features)
        x = self.attn(x)
        x = self.fc(x)

        return torch.sigmoid(x)