import torch
import torch.nn as nn

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


checkpoint_path = "emotion2vec_final_model.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)

model = EmotionClassifier()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


import numpy as np
from funasr import AutoModel

# Load the trained Emotion2Vec+ encoder
encoder = AutoModel(
    model="iic/emotion2vec_plus_base",
    hub="hf",
    device="cpu",
    disable_update=True
)

# Extract embedding from the converted real audio
result = encoder.generate(
    input="converted_audio.wav",
    granularity="utterance",
    extract_embedding=True
)

embedding = np.asarray(
    result[0]["feats"],
    dtype=np.float32
).reshape(1, -1)


# Normalize using the exact training-time scaler parameters
scaler_mean = np.asarray(
    checkpoint["scaler_mean"], dtype=np.float32
)

scaler_scale = np.asarray(
    checkpoint["scaler_scale"], dtype=np.float32
)

embedding_scaled = (embedding - scaler_mean) / scaler_scale


# Convert normalized embedding to PyTorch tensor
input_tensor = torch.from_numpy(embedding_scaled).float()

# Run trained classifier
with torch.no_grad():
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0)

# Get predicted class
predicted_index = int(torch.argmax(probabilities).item())
idx_to_emotion = checkpoint["idx_to_emotion"]
predicted_emotion = idx_to_emotion[predicted_index]

print("\n========== EMOTION PREDICTION ==========")
print("Predicted emotion:", predicted_emotion)
print("\nClass probabilities:")

for index, probability in enumerate(probabilities.tolist()):
    emotion = idx_to_emotion[index]
    print(f"{emotion:10s}: {probability * 100:.2f}%")

print("========================================")