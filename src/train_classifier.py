'''
Goal: Turn each video into one 768-number vector (mean + std of its DINOv2 frame embeddings),
train logistic regression, print training and validation metrics plus confusion matrices,
and save the model.

Usage: python3 src/train_classifier.py
'''

import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay

# Best value for C on validation as determined through experiments with C = 0.01, 0.1, 1, 10
# on mean+std pooling and mean-only
C = 0.01

# Turn one video's (T, 384) embeddings into one (768,) vector:
# the average of each feature and how much each feature changes over the video
def pool(emb):
    return np.concatenate([emb.mean(axis=0), emb.std(axis=0)])

X_train = []
y_train = []
X_validation = []
y_validation = []

with open("data/metadata.json", "r") as fp:
    videos = json.load(fp)

for video in videos:
    if video["split"] == "test":
        continue
    embedding_path = "data/embeddings/" + video["name"].replace(".mp4", ".npz")
    emb = np.load(embedding_path)["emb"]
    x = pool(emb)

    if video["split"] == "train":
        X_train.append(x)
        y_train.append(video["class"])

    elif video["split"] == "valid":
        X_validation.append(x)
        y_validation.append(video["class"])

X_train = np.array(X_train)
X_validation = np.array(X_validation)

print(f"X_train shape: {X_train.shape}")
print(f"X_validation shape: {X_validation.shape}")

# Create enum for class labels e.g bench_pressing -> 0, front_raise -> 1
classes = sorted(set(y_train))
CLASS_LABELS = {name: i for i, name in enumerate(classes)}

# y_train and y_validation are now np.arrays of the exercise numbers
y_train = np.array([CLASS_LABELS[exercise] for exercise in y_train])
y_validation = np.array([CLASS_LABELS[exercise] for exercise in y_validation])

print(f"Classes: {classes}")
print(f"Class counts (training): {np.bincount(y_train)}")
print(f"Class counts (validation): {np.bincount(y_validation)}")

# Standardize X
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation) # no fitting, since we should not have access to validation data stats

# Train Logistic Regression
with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
    model = LogisticRegression(C=C, max_iter=2000)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_validation = model.predict(X_validation)

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
    plt.title(f"Logistic regression: {name}")
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/cm_baseline_{name.lower()}.png", dpi=150)
    plt.close()

evaluate("Train", y_train, pred_train)
evaluate("Validation", y_validation, pred_validation)

# Save everything needed to classify a new video: the model, the TRAINING scaler,
# and the class names (to turn the predicted number back into its exercise name)
os.makedirs("models", exist_ok=True)
joblib.dump({"model": model, "scaler": scaler, "classes": classes}, "models/baseline_logreg.joblib")
print("Saved model to models/baseline_logreg.joblib")
