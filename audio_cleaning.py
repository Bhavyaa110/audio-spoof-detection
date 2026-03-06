import os
import librosa
import noisereduce as nr
import soundfile as sf
from tqdm import tqdm

languages = ["hindi", "punjabi"]
types = ["real", "fake"]
durations = ["3sec", "5sec"]

base_path = "dataset"

for lang in languages:
    for t in types:
        for dur in durations:

            input_folder = os.path.join(base_path, lang, t, "framed", dur)
            output_folder = os.path.join(base_path, lang, t, "cleaned", dur)

            if not os.path.exists(input_folder):
                continue

            os.makedirs(output_folder, exist_ok=True)

            files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]

            for file in tqdm(files, desc=f"{lang}-{t}-{dur}"):

                input_path = os.path.join(input_folder, file)
                output_path = os.path.join(output_folder, file)

                # Load audio
                y, sr = librosa.load(input_path, sr=16000)

                # Noise reduction
                reduced_noise = nr.reduce_noise(y=y, sr=sr)

                # Normalize
                normalized = librosa.util.normalize(reduced_noise)

                # Save cleaned audio
                sf.write(output_path, normalized, sr)