#edit before using change the orignal folder name/paths
import static_ffmpeg
static_ffmpeg.add_paths()

import sys
import audioop
sys.modules['audioop'] = audioop
sys.modules['pyaudioop'] = audioop

import os
from pydub import AudioSegment

base_path = "dataset"
languages = ["hindi", "punjabi"]

# ─── Convert mp3 to wav (keep originals) ──────────────────────────────────────
print(" Converting mp3 to wav...")

for lang in languages:
    original_folder = os.path.join(base_path, lang, "AiGen", "original")

    for file in os.listdir(original_folder):
        if file.endswith(".mp3"):
            mp3_path = os.path.join(original_folder, file)
            wav_path = os.path.join(original_folder, os.path.splitext(file)[0] + ".wav")

            if os.path.exists(wav_path):
                print(f"Skipping (already exists): {file}")
                continue

            audio = AudioSegment.from_mp3(mp3_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(wav_path, format="wav")

            print(f"  Converted: {file} → {os.path.basename(wav_path)}")

print("All conversions done!")