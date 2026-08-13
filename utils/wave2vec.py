import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor


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


        return wav2vec_feat.squeeze(0)