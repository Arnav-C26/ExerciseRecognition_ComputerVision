# Live exercise classifier: watches a webcam (or a video file) and shows which exercise is being performed.
# Every 0.1 s one frame is embedded with DINOv2 (same as embed.py). The last 32 embeddings (3.2 s)
# are fed to the model trained by train_GRU.py or train_classifier.py.
# Usage:
# RUN IN ROOT OF FOLDER
# `python src/live_app.py` (webcam, GRU model)
# `python src/live_app.py --model logreg` (webcam, logistic regression baseline)
# `python src/live_app.py --video data/sample/stu9_64.mp4` (test on a video file instead of the webcam)
# Keys while running: q = quit, r = reset (clear the frame buffer when you switch exercises)
# Needs models/gru.pt (from train_GRU.py) or models/baseline_logreg.joblib (from train_classifier.py)

import sys
import time
import argparse
import collections
import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

TARGET_FPS = 10  
SMOOTHING = 0.7   
MIN_CONFIDENCE = 0.4  
GRU_PATH = "models/gru.pt"
LOGREG_PATH = "models/baseline_logreg.joblib"

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        h = self.dropout(h_n[-1])
        return self.linear(h)

def load_gru(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False) 
    classes, window = ckpt["classes"], ckpt["window"]
    mean, std = ckpt["mean"], ckpt["std"]
    model = GRUClassifier(384, ckpt["hidden"], len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def predict(frames):
        x = (frames - mean) / std  # standardize with the TRAINING mean/std
        x = torch.tensor(x[None], dtype=torch.float32)  # a batch of one window: (1, 32, 384)
        with torch.no_grad():
            return torch.softmax(model(x), dim=1)[0].numpy()

    return classes, window, predict


def load_logreg(path):
    saved = joblib.load(path)
    model, scaler, classes = saved["model"], saved["scaler"], saved["classes"]

    def predict(frames):
        # same pooling as train_classifier.py: mean + std of the frames -> 768 numbers
        x = np.concatenate([frames.mean(axis=0), frames.std(axis=0)])
        x = scaler.transform([x])
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return model.predict_proba(x)[0]

    # the baseline was trained on whole videos, but the app only has a short buffer, so use 32 frames
    return classes, 32, predict


# DINOv2 frame embedding, same as embed.py: keep the CLS token (384 numbers)
def make_embedder():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    encoder = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()

    @torch.no_grad()
    def embed(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) 
        inputs = processor(images=rgb, return_tensors="pt").to(device)
        return encoder(**inputs).last_hidden_state[:, 0].cpu().numpy()[0]

    return embed


# draw the prediction and a bar for each exercise on top of the frame
def draw(frame, classes, probs, n_frames, window):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h - 30 - 28 * len(classes)), (260, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    if probs is None:
        text, color = f"Collecting frames... {n_frames}/{window}", (200, 200, 200)
    else:
        best = int(np.argmax(probs))
        if probs[best] < MIN_CONFIDENCE:
            text, color = "Not sure", (0, 200, 255)
        else:
            text, color = f"{classes[best]}  {probs[best] * 100:.0f}%", (0, 255, 0)
    cv2.putText(frame, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    if probs is not None:
        for i, name in enumerate(classes):
            y = h - 20 - 28 * (len(classes) - 1 - i)
            cv2.rectangle(frame, (110, y - 14), (110 + int(probs[i] * 140), y), (0, 255, 0), -1)
            cv2.putText(frame, name[:9], (8, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gru", "logreg"], default="gru")
    parser.add_argument("--video", help="path to a video file to use instead of the webcam")
    parser.add_argument("--camera", type=int, default=0, help="webcam number (try 1 if 0 is the wrong camera)")
    args = parser.parse_args()

    if args.model == "gru":
        classes, window, predict = load_gru(GRU_PATH)
    else:
        classes, window, predict = load_logreg(LOGREG_PATH)
    print(f"Loaded {args.model} model, classes: {classes}")
    embed = make_embedder()

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        sys.exit("Could not open the video source. On a Mac, allow camera access for your terminal/editor "
                 "in System Settings > Privacy & Security > Camera.")

    # for a video file, keep 1 out of every `step` frames (like embed.py)
    # for a webcam, keep a frame every 1/10 s of real time
    video_fps = cap.get(cv2.CAP_PROP_FPS) if args.video else 0
    step = max(round(video_fps / TARGET_FPS), 1) if args.video else 1
    delay = max(int(1000 / video_fps), 1) if args.video and video_fps > 0 else 1

    buffer = collections.deque(maxlen=window) 
    smooth = None                    
    last_taken = 0.0
    frame_no = -1

    failed = 0 
    while True:
        ret, frame = cap.read()
        if not ret:
            if args.video:
                break  
            # webcam: the first reads often fail while the camera warms up (or permission is blocked),
            # so keep retrying for about 5 seconds before giving up
            failed += 1
            if failed > 100:
                sys.exit("The camera opened but gave no frames. On a Mac, allow camera access for VS Code "
                         "(or Terminal) in System Settings > Privacy & Security > Camera, then fully quit and "
                         "reopen it. Also close other apps using the camera, or try --camera 1.")
            cv2.waitKey(50)
            continue
        failed = 0
        frame_no += 1

        now = time.time()
        if args.video:
            take = frame_no % step == 0
        else:
            take = now - last_taken >= 1 / TARGET_FPS
        if take:
            last_taken = now
            buffer.append(embed(frame)) 
            if len(buffer) == window:
                probs = predict(np.stack(buffer))
                smooth = probs if smooth is None else SMOOTHING * smooth + (1 - SMOOTHING) * probs

        shown = frame if args.video else cv2.flip(frame, 1)
        cv2.imshow("Exercise classifier (q = quit, r = reset)", draw(shown, classes, smooth, len(buffer), window))

        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            buffer.clear()
            smooth = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
