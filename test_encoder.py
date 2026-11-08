import torch
from funasr import AutoModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Loading Emotion2Vec+ Base...")
print("Device:", device)

emotion2vec_model = AutoModel(
    model="iic/emotion2vec_plus_base",
    hub="hf",
    device=device,
    disable_update=True
)

print("\nSUCCESS: Emotion2Vec+ Base loaded!")