'''
Demo: classify one validation video with the trained GRU and compare to the true label.

Usage (from the repo root): python3 src/demo.py
Downloads models/gru.pt from Google Drive the first time if it's missing.
'''

import os
import sys
import json
import urllib.request
import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

VIDEO = "data/sample/stu10_67.mp4" # a sample from the validation set
MODEL_PATH = "models/gru.pt"
DRIVE_FILE_ID = "1g8SULhIQ1xVcuFJhZg2vvp6ibYpQ1cJV" # from Google drive
TARGET_FPS = 10

# Download the weights if they're missing
if not os.path.exists(MODEL_PATH):
    if DRIVE_FILE_ID == "PASTE_ID_HERE":
        sys.exit(f"See README.md. {MODEL_PATH} is missing and no Google Drive link is set.")
    print("Downloading model weights from Google Drive...")
    os.makedirs("models", exist_ok=True)
    urllib.request.urlretrieve(f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}", MODEL_PATH)

# Copy GRUClassifier model (directly from train_GRU.py) and load the trained weights
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        h = h_n[-1]
        h = self.dropout(h)
        scores = self.linear(h)
        return scores

dump = torch.load(MODEL_PATH, weights_only=False) # the dict saved by train_GRU.py
classes = dump["classes"]
WINDOW = dump["window"]

# Rebuild the model
model = GRUClassifier(384, dump["hidden"], len(classes))
model.load_state_dict(dump["state_dict"])
model.eval() # turn off dropoff

# Read the video at 10 fps (from embed.py)
cap = cv2.VideoCapture(VIDEO)
step = max(round(cap.get(cv2.CAP_PROP_FPS) / TARGET_FPS), 1) # e.g 30 / 10 = 3, so keep 1 frame out of 3
frames = []
i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if i % step == 0:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    i += 1
cap.release()

print(f"Read {len(frames)} frames from {VIDEO}")
#  DINOv2 embedding for every kept frame -> (T, 384) (from embed.py)
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
dinov2 = AutoModel.from_pretrained("facebook/dinov2-small").eval()

emb = []
with torch.no_grad():
    for start in range(0, len(frames), 32): # 32 frames at a time
        inputs = processor(images=frames[start:start + 32], return_tensors="pt")
        cls = dinov2(**inputs).last_hidden_state[:, 0] # the CLS token: 384 numbers per frame
        emb.append(cls.numpy())
emb = np.concatenate(emb)
print(f"Embeddings: {emb.shape}")

# cut into windows, average the probabilities
# from predict_video in train_GRU.py
STRIDE = 16
video = (emb - dump["mean"]) / dump["std"]
starts = range(0, len(video) - WINDOW + 1, STRIDE) # e.g 0, 16, 32
windows = torch.tensor(np.stack([video[s:s + WINDOW] for s in starts]), dtype=torch.float32)

with torch.no_grad():
    probs = torch.softmax(model(windows), dim=1) # (num_windows, 7)

probs = probs.mean(dim=0).numpy() # (7,) the answer for the whole video

# Get the true label
with open("data/metadata.json", "r") as fp:
    videos = json.load(fp)
true_label = next(video["class"] for video in videos if video["name"] == os.path.basename(VIDEO))

# Print res
predicted = classes[probs.argmax()]
print(f"\nVideo: {VIDEO} ({len(windows)} windows of {WINDOW} frames)")
print(f"True exercise: {true_label}")
print(f"Predicted exercise: {predicted} ({probs.max() * 100:.0f}% confident)")

print("\nProbabilities:")
for i in np.argsort(probs)[::-1]: # sort by highest
    print(f"  {classes[i]:<15} {probs[i] * 100:5.1f}%")
