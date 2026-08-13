import os
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset

from utils.beats import FeatureExtractor
#from utils.wave2vec import FeatureExtractor
#from utils.beats_wave2vec import FeatureExtractor
from config import DEVICE


class MixDataset(Dataset):

    def __init__(self, split):

        parquet_map = {
            "train": "train.parquet",
            "val": "validation.parquet",
            "test": "test.parquet"
        }

        parquet_path = os.path.join(
            "data_repo",
            "speaker_split",
            parquet_map[split]
        )

        self.df = pd.read_parquet(
            parquet_path
        )

        self.audio_root = "data_repo"

        self.extractor = FeatureExtractor(
            DEVICE
        )

        print(
            f"Parquet : {parquet_path}"
        )
        print(
            f"Audio Root : {self.audio_root}"
        )

    def __len__(self):
        return len(
            self.df
        )

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        rel_path = row["path"].replace(
            "/",
            os.sep
        )

        audio_path = os.path.join(
            self.audio_root,
            rel_path
        )

        if not os.path.exists(
            audio_path
        ):
            raise FileNotFoundError(
                audio_path
            )

        wav, sr = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )

        wav = torch.tensor(
            wav,
            dtype=torch.float32
        ).unsqueeze(0)

        feat = self.extractor.extract(
            wav,
            sr
        )

        if row["label"] == "real":
            label = 0
        elif row["label"] == "fake":
            label = 1
        else:
            raise ValueError(f"Unknown label: {row['label']}")
        return feat, label