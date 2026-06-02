import torch
import torchaudio
from transformers import Wav2Vec2Model
from transformers import Wav2Vec2Processor
from transformers import AutoProcessor
from transformers import AutoModel

class FeatureExtractor:

    def __init__(self, device):

        self.device = device

        print("Loading wav2vec...")

        self.w2v_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base"
        )

        self.w2v = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        ).to(device)

        print("Loading BEATs...")

        self.beats_processor = AutoProcessor.from_pretrained(
            "facebook/beats-base"
        )

        self.beats = AutoModel.from_pretrained(
            "facebook/beats-base"
        ).to(device)

    def extract(self, waveform, sr):

        if sr != 16000:
            waveform = torchaudio.functional.resample(
                waveform,
                sr,
                16000
            )

        waveform = waveform.squeeze()

        # WAV2VEC
        w2v_inputs = self.w2v_processor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )

        # BEATS
        beats_inputs = self.beats_processor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )

        with torch.no_grad():

            w2v_feat = self.w2v(
                w2v_inputs.input_values.to(self.device)
            ).last_hidden_state

            beats_feat = self.beats(
                beats_inputs.input_values.to(self.device)
            ).last_hidden_state

        # concat

        feat = torch.cat(
            [w2v_feat, beats_feat],
            dim=-1
        )

        return feat.squeeze(0)