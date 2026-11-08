import numpy as np
from funasr import AutoModel

print("Loading Emotion2Vec+ Base...")

model = AutoModel(
    model="iic/emotion2vec_plus_base",
    hub="hf",
    device="cpu",
    disable_update=True
)

audio_path = "converted_audio.wav"

print("Extracting embedding...")

result = model.generate(
    input=audio_path,
    granularity="utterance",
    extract_embedding=True
)

embedding = np.asarray(
    result[0]["feats"],
    dtype=np.float32
).reshape(1, -1)

print("Embedding extraction successful!")
print("Embedding shape:", embedding.shape)
print("Data type:", embedding.dtype)
print("NaN values:", np.isnan(embedding).sum())
print("Infinite values:", np.isinf(embedding).sum())