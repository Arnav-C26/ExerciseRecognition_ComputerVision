# Extract per-frame DINOv2 embeddings for each video at 10 fps
# Usage:
# RUN IN ROOT OF FOLDER
# `python src/embed.py data/sample/squat.mp4` (one video)
# OR 
# `python src/embed.py` (for all videos in data/metadata.json, takes a while)

import sys
import os
import json
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

TARGET_FPS = 10
BATCH_SIZE = 32

# dinov2 by Meta is a small pretrained img classifier already trained on around 142 million samples
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small").eval() 

def read_frames(path):
    cap = cv2.VideoCapture(path)
    # How many frames to skip
    # Frame rate / target fps
    # e.g 30 / 10 = 3, so keep 1 frame out of 3 frames
    step = max(round(cap.get(cv2.CAP_PROP_FPS) / TARGET_FPS), 1)
    frames = [] # pictures to be kept
    idx = [] # frame numbers of those pictures
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # only save 1 / step frames
        if i % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx.append(i) # mark the idx of that image
        i += 1
    cap.release()

    return frames, np.array(idx)


'''
load list of imgs
feed to dinov2 in groups of 32
get back 384 numbers per img (embedding)
'''
@torch.no_grad()
def embed(frames: list) -> np.ndarray:
    res = []
    for start in range(0, len(frames), BATCH_SIZE):
        batch = frames[start:start + BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt")

        outputs = model(**inputs)
        # last_hidden_state: [batch, tokens, dim]
        feats = outputs.last_hidden_state

        cls = feats[:, 0]
        res.append(cls.cpu().numpy())

    # stack all batches into one [num_frames, dim] array
    return np.concatenate(res)


if len(sys.argv) > 1:
    paths = [sys.argv[1]]
else:
    paths = [video["path"] for video in json.load(open("data/metadata.json"))]

os.makedirs("data/embeddings", exist_ok=True)

for n, path in enumerate(paths):
    out = "data/embeddings/" + os.path.basename(path).replace(".mp4", ".npz")
    if os.path.exists(out):
        continue
    frames, idx = read_frames(path)
    if len(frames) == 0:
        print(f"could not read {path}, skipping")
        continue
    np.savez(out, emb=embed(frames), frame_idx=idx)
    print(f"[{n + 1}/{len(paths)}] {path}")