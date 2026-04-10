import os
import torch
from torch.utils.data import Dataset
from utils.audio_utils import load_audio

class ASVspoofDataset(Dataset):
    def __init__(self, protocol_file, data_dir):
        self.data = []
        self.data_dir = data_dir

        with open(protocol_file, "r") as f:
            for line in f:
                parts = line.strip().split()

                file_name = parts[1]
                label = parts[-1]

                label = 0 if label == "bonafide" else 1

                self.data.append((file_name, label))
        
        self.data = self.data[:10000]  # Limit to 500 samples for faster training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name, label = self.data[idx]

        path = os.path.join(self.data_dir, file_name + ".flac")

        audio = load_audio(path)

        return torch.tensor(audio).unsqueeze(0), torch.tensor(label)