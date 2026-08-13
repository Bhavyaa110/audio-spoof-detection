import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from unilm.beats.BEATs import BEATs, BEATsConfig

import pickle

class FeatureExtractor:

    def __init__(self, device):

        self.device = device

        print("Loading wav2vec...")

        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base"
        )

        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        ).to(device)

        self.wav2vec.eval()

        print("Loading BEATs...")

        checkpoint = torch.load(
            "checkpoints/BEATs_iter3_plus_AS2M.pt",
            map_location=device
        )

        cfg = BEATsConfig(
            checkpoint["cfg"]
        )

        self.beats = BEATs(
            cfg
        )

        self.beats.load_state_dict(
            checkpoint["model"]
        )

        self.beats = self.beats.to(
            device
        )

        self.beats.eval()
        with open(
            "checkpoints/pca_mix.pkl",
            "rb"
        ) as f:

            self.pca = pickle.load(f)

    def extract(self, wav, sr):

        wav = wav.to(
            self.device
        )

        if sr != 16000:

            wav = F.interpolate(
                wav.unsqueeze(0),
                size=int(
                    wav.shape[-1]
                    * 16000
                    / sr
                ),
                mode="linear",
                align_corners=False
            ).squeeze(0)

        with torch.no_grad():

            # WAV2VEC
            inp = self.processor(
                wav.squeeze(0).cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            )

            inp = {
                k: v.to(self.device)
                for k, v in inp.items()
            }

            wav2vec_feat = self.wav2vec(
                **inp
            ).last_hidden_state

            # BEATS
            beats_feat, _ = self.beats.extract_features(
                wav
            )

            # ALIGN LENGTHS
            min_len = min(
                wav2vec_feat.size(1),
                beats_feat.size(1)
            )

            wav2vec_feat = wav2vec_feat[
                :, :min_len, :
            ]

            beats_feat = beats_feat[
                :, :min_len, :
            ]

            # CONCAT
            feat = torch.cat(
                [
                    wav2vec_feat,
                    beats_feat
                ],
                dim=2
            )

            feat = feat.squeeze(0)

            feat = self.pca.transform(
                feat.cpu().numpy()
            )

            feat = torch.tensor(
                feat,
                dtype=torch.float32,
                device=self.device
            )

            return feat