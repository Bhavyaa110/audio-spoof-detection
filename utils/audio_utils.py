import librosa
import numpy as np
from config import *

def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    if len(audio) > AUDIO_LEN:
        audio = audio[:AUDIO_LEN]
    else:
        pad = AUDIO_LEN - len(audio)
        audio = np.pad(audio, (0, pad))

    return audio.astype(np.float32)