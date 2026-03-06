import os
from pydub import AudioSegment

base_path = "dataset"

languages = ["hindi", "punjabi"]
classes = ["real", "fake"]

frame_sizes = {
    "3sec": 3000,
    "5sec": 5000
}

for lang in languages:
    for cls in classes:

        original_folder = os.path.join(base_path, lang, cls, "original")
        framed_folder = os.path.join(base_path, lang, cls, "framed")

        folder_3 = os.path.join(framed_folder, "3sec")
        folder_5 = os.path.join(framed_folder, "5sec")

        # ---------- Delete old WAV files ----------
        for f in os.listdir(original_folder):
            if f.endswith(".wav"):
                os.remove(os.path.join(original_folder, f))

        # ---------- Delete old frames ----------
        for folder in [folder_3, folder_5]:
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    os.remove(os.path.join(folder, f))

        # ---------- Convert and Frame ----------
        for file in os.listdir(original_folder):
            file_path = os.path.join(original_folder, file)

            if file.endswith(".mp4") or file.endswith(".m4a"):
                audio = AudioSegment.from_file(file_path)
                audio = audio.set_frame_rate(16000).set_channels(1)
                name = os.path.splitext(file)[0]
                wav_path = os.path.join(original_folder, name + ".wav")

                # convert to wav
                audio.export(wav_path, format="wav")

                # Framing
                for label, frame_size in frame_sizes.items():
                    for i in range(0, len(audio), frame_size):
                        frame = audio[i:i+frame_size]

                        if len(frame) == frame_size:
                            frame_name = f"{name}_{i//frame_size}.wav"

                            if label == "3sec":
                                save_path = os.path.join(folder_3, frame_name)
                            else:
                                save_path = os.path.join(folder_5, frame_name)

                            frame.export(save_path, format="wav")

print("\nALL DONE.")