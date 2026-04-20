import os
import librosa
import soundfile as sf
import numpy as np
from TTS.api import TTS

# ===== CONFIG =====
dataset_path = r"C:\Users\rajgu\audio-spoof-detection\dataset\hindi\real\framed\5sec\Mozilla_dataset"   # or your dataset root
output_path = r"C:\Users\rajgu\audio-spoof-detection\dataset\hindi\AiGen\original\xtts\Mozilla_outputs"
os.makedirs(output_path, exist_ok=True)

# 🔥 FULL TEXT
full_text = """आज सुबह कमल और खालिद घर के पास खड़े थे। गगन में घने बादल थे और ठंडी हवा चल रही थी। कमल ने कहा, तल और थल अलग होते हैं, दल और ढल का भी फर्क होता है, पल और फल एक जैसे नहीं होते। गीता और गोपाल ने धीरे से ध्यान दिया और साफ आवाज में दोहराया। बच्चे बोले, बल और भाल, बर और भर, कर और खर, सब अलग सुनाई देते हैं। बाज़ार में फैज़ान, ज़हीर और क़ासिम अपनी दुकान पर खड़े थे। फूल, फल, सब्ज़ी और गरम पकवान की खुशबू आ रही थी। सबने अन, अंग, अम, अंध और आह जैसे शब्दों का स्पष्ट उच्चारण किया। अंत में सबने तेज, मध्यम और धीमी गति में वही वाक्य फिर से बोला।"""

# ===== SPLIT INTO CHUNKS (<=150 chars) =====
def split_text(text, max_len=150):
    sentences = text.split("।")
    chunks = []
    current = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        if len(current) + len(s) < max_len:
            current += s + "। "
        else:
            chunks.append(current.strip())
            current = s + "। "

    if current:
        chunks.append(current.strip())

    return chunks

text_chunks = split_text(full_text)

print("Chunks:")
for i, c in enumerate(text_chunks):
    print(i, len(c))

# ===== LOAD MODEL =====
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

# ===== MERGE AUDIO FUNCTION =====
def create_reference_audio(files, out_path):
    audios = []
    for f in files:
        try:
            audio, sr = librosa.load(f, sr=22050)
            audio = librosa.util.normalize(audio)
            audios.append(audio)
        except:
            continue

    if not audios:
        return None

    combined = np.concatenate(audios)
    sf.write(out_path, combined, 22050)
    return out_path

# ===== CONCATENATE OUTPUT CHUNKS =====
def merge_outputs(audio_files, out_path):
    audios = []
    for f in audio_files:
        audio, sr = librosa.load(f, sr=22050)
        audios.append(audio)

    final_audio = np.concatenate(audios)
    sf.write(out_path, final_audio, 22050)

# ===== MAIN LOOP =====
for speaker in os.listdir(dataset_path):
    speaker_folder = os.path.join(dataset_path, speaker)

    if not os.path.isdir(speaker_folder):
        continue

    print(f"\nProcessing speaker: {speaker}")

    wav_files = [
        os.path.join(speaker_folder, f)
        for f in os.listdir(speaker_folder)
        if f.endswith(".wav")
    ]

    if len(wav_files) < 3:
        print("Skipping (not enough data)")
        continue

    wav_files = wav_files[:30]  # limit

    ref_path = os.path.join(output_path, f"{speaker}_ref.wav")
    ref_audio = create_reference_audio(wav_files, ref_path)

    if ref_audio is None:
        continue

    chunk_outputs = []

    # ===== GENERATE PER CHUNK =====
    for i, chunk in enumerate(text_chunks):
        chunk_file = os.path.join(output_path, f"{speaker}_chunk_{i}.wav")

        tts.tts_to_file(
            text=chunk,
            speaker_wav=ref_audio,
            language="hi",
            temperature=0.7,
            top_k=50,
            top_p=0.85,
            repetition_penalty=2.0,
            file_path=chunk_file
        )

        chunk_outputs.append(chunk_file)

    # ===== MERGE FINAL OUTPUT =====
    final_output = os.path.join(output_path, f"{speaker}_final.wav")
    merge_outputs(chunk_outputs, final_output)

    print(f"✅ Done: {speaker}")

print("🚀 All speakers processed with full text!")

