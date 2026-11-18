'''
Goal: Train a GRU on 32-frame windows of each video's DINOv2 embeddings.
This may be better that a logistic regression classifier, in that it'll take into account motion (order of frames)
Prints training and validation metrics plus confusion matrices, compares against the
logistic regression baseline on the same 32-frame windows, and saves the model.
The higher result should be the model used for the live app.

Usage: python3 src/train_GRU.py
(MAKE SURE TO RUN `src/train_classifier.py` FIRST, so models/baseline_logreg.joblib exists for the comparison)
'''

import os
import json
import copy
import joblib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay


torch.manual_seed(42)
np.random.seed(42)

WINDOW = 32 # Number of frames per window
HIDDEN = 128

class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        video = self.X[i]
        rand_start_frame = np.random.randint(0, len(video) - WINDOW + 1)
        # slice out WINDOW frames: shape (32, 384)
        frames = video[rand_start_frame: rand_start_frame + WINDOW]

        return torch.tensor(frames, dtype=torch.float32), torch.tensor(self.y[i])

X_train = [] # 639 arrays (videos) each with their own length
y_train = []
X_validation = []
y_validation = []

with open("data/metadata.json", "r") as fp:
    videos = json.load(fp)

for video in videos:
    if video["split"] == "test":
        continue
    emb = np.load("data/embeddings/" + video["name"].replace(".mp4", ".npz"))["emb"]

    if video["split"] == "train":
        X_train.append(emb)
        y_train.append(video["class"])

    elif video["split"] == "valid":
        X_validation.append(emb)
        y_validation.append(video["class"])

print(f"X_train shape: {X_train[0].shape}")
print(f"X_validation shape: {X_validation[0].shape}")

# Create enum for class labels e.g bench_pressing -> 0, front_raise -> 1
classes = sorted(set(y_train))
CLASS_LABELS = {name: i for i, name in enumerate(classes)}

# y_train and y_validation are now np.arrays of the exercise numbers
y_train = np.array([CLASS_LABELS[exercise] for exercise in y_train])
y_validation = np.array([CLASS_LABELS[exercise] for exercise in y_validation])

print(f"Classes: {classes}")
print(f"Class counts (training): {np.bincount(y_train)}")
print(f"Class counts (validation): {np.bincount(y_validation)}")

# Keep the raw (unstandardized) validation videos for the baseline comparison at the end
X_validation_raw = X_validation

# Standardize
all_train_frames = np.concatenate(X_train)
mean = all_train_frames.mean(axis=0)
std = all_train_frames.std(axis=0) + 1e-6 # to avoid divide by 0

X_train = [(x - mean) / std for x in X_train]
X_validation = [(x - mean) / std for x in X_validation]

train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=16, shuffle=True)

# Define Model
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, 32, 384), a batch of windows

        _, h_n = self.gru(x) # h_n is the GRU's memory after the last frame: (1, batch, 128)

        h = h_n[-1] # (batch, 128)

        h = self.dropout(h) # apply dropout at 0.3

        scores = self.linear(h)

        return scores


model = GRUClassifier(input_size=384, hidden_size=HIDDEN, num_classes=len(classes))
print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

loss_fn = nn.CrossEntropyLoss() # how wrong the 7 scores are vs the true label

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Predict one whole video
# 1) cut it into overlapping windows
# 2) predict each
# 3) average probabilities
def predict_video(model, video, stride=16):
    starts = range(0, len(video) - WINDOW + 1, stride) # e.g 0, 16, 32
    windows = torch.tensor(np.stack([video[s:s + WINDOW] for s in starts]), dtype=torch.float32)

    with torch.no_grad():
        probs = torch.softmax(model(windows), dim=1) # (num_windows, 7)

    return probs.mean(dim=0) # (7,) the answer for the whole video

# Train
EPOCHS = 60
best_acc = 0

for epoch in range(EPOCHS):
    # One pass over all training videos
    model.train()
    total_loss = 0
    for windows, labels in train_loader:
        optimizer.zero_grad() # clear the previous batch's gradients
        scores = model(windows) # forward: (16, 32, 384) -> (16, 7)
        loss = loss_fn(scores, labels)
        loss.backward() # back prop
        optimizer.step() # update weights at each gradient
        total_loss += loss.item()

    model.eval()
    preds = [predict_video(model, video).argmax().item() for video in X_validation]
    valid_acc = np.mean(np.array(preds) == y_validation)

    print(f"epoch {epoch + 1:2d}  train loss {total_loss / len(train_loader):.3f}  valid acc {valid_acc:.3f}")

    if valid_acc > best_acc:
        best_acc = valid_acc
        best_epoch = epoch + 1
        best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
print(f"Best validation accuracy {best_acc:.3f} at epoch {best_epoch}")

# Predict every train and validation video with the best weights
model.eval()
pred_train = np.array([predict_video(model, video).argmax().item() for video in X_train])
pred_validation = np.array([predict_video(model, video).argmax().item() for video in X_validation])

# Evaluate
def evaluate(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Macro F1-Score: {f1_score(y_true, y_pred, average='macro'):.3f}")
    print(classification_report(y_true, y_pred, target_names=classes))

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=classes,
        xticks_rotation=45,
        normalize="true",
        values_format=".2f",
        colorbar=False,
    )
    plt.title(f"GRU: {name}")
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/cm_gru_{name.lower()}.png", dpi=150)
    plt.close()

evaluate("Train", y_train, pred_train)
evaluate("Validation", y_validation, pred_validation)

# Window accuracy: test both models on short clips instead of whole videos.
# Slide a 32-frame window (3.2 s) across every validation video and count how often
# the GRU and the logistic regression baseline each guess the exercise right.

# The higher result should indicate the model we are going to use for the live app
baseline = joblib.load("models/baseline_logreg.joblib")
baseline_model = baseline["model"]
baseline_scaler = baseline["scaler"]

gru_correct = 0
baseline_correct = 0
num_windows = 0

for i in range(len(X_validation)):
    video = X_validation[i] # standardized, for the GRU
    video_raw = X_validation_raw[i] # not standardized, for the baseline (it uses its own scaler)
    label = y_validation[i]

    for start in range(0, len(video) - WINDOW + 1, 16):
        end = start + WINDOW

        # GRU prediction
        window = torch.tensor(np.array([video[start:end]]), dtype=torch.float32) # a batch of one: (1, 32, 384)
        with torch.no_grad():
            scores = model(window)
        if scores.argmax().item() == label:
            gru_correct += 1

        # Baseline guess for this window: mean + std of the 32 frames, then its scaler
        window_raw = video_raw[start:end]
        x = np.concatenate([window_raw.mean(axis=0), window_raw.std(axis=0)])
        x = baseline_scaler.transform([x])
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            prediction = baseline_model.predict(x)[0]
        if prediction == label:
            baseline_correct += 1

        num_windows += 1

print("GRU vs. Logistic Regression Window Accuracy")
print(f"Number of windows: {num_windows}")
print(f"GRU: {gru_correct / num_windows:.3f}")
print(f"Logistic regression: {baseline_correct / num_windows:.3f}")
choice = "GRU" if gru_correct / num_windows >= baseline_correct / num_windows else "Logistic Regression"
print(f"We should use {choice} as the classifier for the live app.")

# Save everything needed to classify a new video: the weights, the TRAINING mean/std,
# the class names, and the settings needed to rebuild the model
os.makedirs("models", exist_ok=True)
torch.save({
    "state_dict": model.state_dict(),
    "mean": mean,
    "std": std,
    "classes": classes,
    "window": WINDOW,
    "hidden": HIDDEN,
}, "models/gru.pt")
print("Saved model to models/gru.pt")
