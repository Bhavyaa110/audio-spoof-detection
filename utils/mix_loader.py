import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from utils.beats_wave2vec import FeatureExtractor
from config import DEVICE

class MixDataset(Dataset):

    def __init__(self, split):

        self.dataset = load_dataset(
            "mandalorian180605/Audio_spoof_detection_dataset",
            split=split
        )

        self.extractor = FeatureExtractor(
            DEVICE
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]

        waveform = torch.tensor(
            audio
        ).unsqueeze(0)

        feat = self.extractor.extract(
            waveform,
            sr
        )

        label = sample["label"]

        return feat, label