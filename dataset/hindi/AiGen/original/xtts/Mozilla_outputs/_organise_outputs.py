import os
import shutil

# ===== PATHS =====
input_folder = r"C:\Users\rajgu\audio-spoof-detection\dataset\hindi\AiGen\original\xtts\Mozilla_outputs"
base_output = r"C:\Users\rajgu\audio-spoof-detection\dataset\hindi\AiGen\original\xtts\Mozilla_outputs"

segmented_folder = os.path.join(base_output, "segmented_cloned")
joined_folder = os.path.join(base_output, "joined_cloned")

os.makedirs(segmented_folder, exist_ok=True)
os.makedirs(joined_folder, exist_ok=True)

# ===== PROCESS =====
for file in os.listdir(input_folder):

    if not file.endswith(".wav"):
        continue

    file_path = os.path.join(input_folder, file)

    # 🔹 SEGMENTED FILES (any number of chunks)
    if "_chunk_" in file:
        speaker, chunk_part = file.split("_chunk_")
        chunk_id = chunk_part.replace(".wav", "")

        speaker_dir = os.path.join(segmented_folder, speaker)
        os.makedirs(speaker_dir, exist_ok=True)

        # clean filename
        new_filename = f"chunk_{chunk_id}.wav"
        new_path = os.path.join(speaker_dir, new_filename)

        shutil.move(file_path, new_path)

    # 🔹 FINAL FILE
    elif "_final" in file:
        speaker = file.replace("_final.wav", "")
        new_path = os.path.join(joined_folder, f"{speaker}.wav")

        shutil.move(file_path, new_path)

    # 🔹 DELETE REF FILES
    elif "_ref" in file:
        os.remove(file_path)

print("✅ Organized all speakers with multiple segments!")