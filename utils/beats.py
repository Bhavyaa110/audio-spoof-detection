import torch
import torch.nn.functional as F

from unilm.beats.BEATs import BEATs, BEATsConfig


class FeatureExtractor:

    def __init__(self, device):

        self.device = device

        print("Loading BEATs...")

        checkpoint = torch.load(
            "checkpoints/BEATs_iter3_plus_AS2M.pt",
            map_location=device
        )

        cfg = BEATsConfig(
            checkpoint["cfg"]
        )

        self.beats = BEATs(cfg)

        self.beats.load_state_dict(
            checkpoint["model"]
        )

        self.beats = self.beats.to(device)

        self.beats.eval()

    def extract(self, wav, sr):

        wav = wav.to(self.device)

        if sr != 16000:

            wav = F.interpolate(
                wav.unsqueeze(0),
                size=int(
                    wav.shape[-1] * 16000 / sr
                ),
                mode="linear",
                align_corners=False
            ).squeeze(0)

        with torch.no_grad():

            beats_feat, _ = self.beats.extract_features(
                wav
            )

        return beats_feat.squeeze(0)