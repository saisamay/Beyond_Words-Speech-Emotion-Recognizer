import os
import sys
import tempfile
import subprocess
import numpy as np
import torch
import torch.nn as nn
from funasr import AutoModel


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CHECKPOINT_PATH = "emotion2vec_final_model.pth"


# --------------------------------------------------
# CLASSIFIER ARCHITECTURE
# Must match the trained architecture exactly
# --------------------------------------------------

class EmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 7)
        )

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------
# AUDIO CONVERSION
# --------------------------------------------------

def convert_to_wav(input_path, output_path):

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        output_path
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )


# --------------------------------------------------
# MAIN INFERENCE
# --------------------------------------------------

def predict_emotion(input_path):

    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Audio/video file not found: {input_path}"
        )

    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    with tempfile.TemporaryDirectory() as temp_dir:

        wav_path = os.path.join(temp_dir, "input.wav")

        print("\n[1/5] Converting audio to 16 kHz mono WAV...")

        convert_to_wav(input_path, wav_path)

        print("[2/5] Loading Emotion2Vec+ Base encoder...")

        encoder = AutoModel(
            model="iic/emotion2vec_plus_base",
            hub="hf",
            device="cpu",
            disable_update=True
        )

        print("[3/5] Extracting 768-D audio embedding...")

        result = encoder.generate(
            input=wav_path,
            granularity="utterance",
            extract_embedding=True
        )

        embedding = np.asarray(
            result[0]["feats"],
            dtype=np.float32
        ).reshape(1, -1)

        if embedding.shape != (1, 768):
            raise ValueError(
                f"Unexpected embedding shape: {embedding.shape}"
            )

        if not np.isfinite(embedding).all():
            raise ValueError("Embedding contains NaN or Inf values.")

        print("[4/5] Loading trained classifier and scaler...")

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False
        )

        model = EmotionClassifier()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        mean = np.asarray(
            checkpoint["scaler_mean"],
            dtype=np.float32
        )

        scale = np.asarray(
            checkpoint["scaler_scale"],
            dtype=np.float32
        )

        if mean.shape != (768,) or scale.shape != (768,):
            raise ValueError("Invalid saved scaler dimensions.")

        if not np.isfinite(scale).all() or np.any(scale == 0):
            raise ValueError("Invalid scaler values.")

        embedding_scaled = (embedding - mean) / scale

        input_tensor = torch.from_numpy(
            embedding_scaled
        ).float()

        print("[5/5] Predicting emotion...")

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(
                logits, dim=1
            ).squeeze(0)

        predicted_index = int(
            torch.argmax(probabilities).item()
        )

        idx_to_emotion = checkpoint["idx_to_emotion"]

        predicted_emotion = idx_to_emotion[predicted_index]

        print("\n" + "=" * 45)
        print("       SPEECH EMOTION RECOGNITION")
        print("=" * 45)

        print(f"\nPredicted Emotion: {predicted_emotion.upper()}")

        print("\nEmotion Probability Distribution:")

        for index, probability in enumerate(
            probabilities.tolist()
        ):
            emotion = idx_to_emotion[index]
            bar = "█" * int(probability * 30)

            print(
                f"{emotion:10s} "
                f"{probability * 100:6.2f}%  {bar}"
            )

        print("\n" + "=" * 45)


# --------------------------------------------------
# PROGRAM ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    print("\nSpeech Emotion Recognition")
    print("-" * 35)

    input_path = input(
        "Enter audio/video file path: "
    ).strip().strip('"').strip("'")

    try:
        predict_emotion(input_path)

    except subprocess.CalledProcessError as error:
        print("\nFFmpeg conversion failed.")
        print(error.stderr.decode(errors="replace"))

    except Exception as error:
        print(f"\nPrediction failed: {error}")
        sys.exit(1)