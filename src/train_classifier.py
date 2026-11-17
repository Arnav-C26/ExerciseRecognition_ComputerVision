'''
Goal: Turn each video into one 384-number vector, train logistic regression, 
and print training and validation metrics plus confusion matricies.

Usage: python3 src/train_baseline.py
'''

import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
X_train = []
y_train = []
X_validation = []
y_validation = []

with open("data/metadata.json", "r") as fp:
    data =json.load(fp)
    for video in data:
        if video["split"] == "test":
            continue
        embedding_path = "data/embeddings/" + video["name"].replace(".mp4", ".npz")
        emb = np.load(embedding_path)["emb"]
        x = emb.mean(axis = 0) # average down the emb to get vec that represents the whole video.

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

# Reassign y_train
# y train is now a np.array of the exercise corresponding number
to_num = [CLASS_LABELS[exercise] for exercise in y_train]
y_train = np.array(to_num)

print(f"Classes: {classes}")
print(f"First 10 (training): {y_train[:10]}")
print(f"Class counts (training): {np.bincount(y_train)}\n")

# Same thing with validation

y_validation = np.array([CLASS_LABELS[exercise] for exercise in y_validation])
print(f"First 10 (validation): {y_validation[:10]}")
print(f"Class counts (validation): {np.bincount(y_validation)}\n")

# Standardize X
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation) # no fitting, since we should not have access to validation data stats

print(X_train.mean(axis=0)[:5].round(3), X_train.std(axis=0)[:5].round(3))
print(X_validation.mean(axis=0)[:5].round(3), X_validation.std(axis=0)[:5].round(3))


# Train Logistic Regression

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_validation = model.predict(X_validation)


# Evaluate
def evaluate(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Macro F1-Score: {f1_score(y_train, pred_train, average='macro')}")
    print(classification_report(y_train, pred_train, target_names=classes))
evaluate("Train", y_train, pred_train)
evaluate("Validation", y_validation, pred_validation)