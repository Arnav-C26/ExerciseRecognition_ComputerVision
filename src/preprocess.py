# Parse RepCount annotations
# Keep only gym exercises
# Save metadata and class count plot.

# RUN IN ROOT OF FOLDER
# Usage: `python3 src/preprocess.py data/LLSP` 

import sys
import os
import json
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Dataset spells some classes differently and has typos e.g "squant"
# so map every spelling to one clean name
# no 'pommel horse' 'battle_rope' or 'others'
# battle rope is pretty imbalanced, with 0 test videos
CLASS_NAMES = {
    "bench_pressing": "bench_pressing", "benchpressing": "bench_pressing",
    "front_raise": "front_raise", "frontraise": "front_raise",
    "pull_up": "pull_up", "pullups": "pull_up",
    "push_up": "push_up", "pushups": "push_up",
    "situp": "situp", "situps": "situp",
    "squat": "squat", "squant": "squat",
    "jump_jack": "jump_jack", "jumpjacks": "jump_jack",
}
if len(sys.argv) > 1:
    root = sys.argv[1]
else:
    root = "data/LLSP"


videos = []
skipped = set()

for split in ["train", "valid", "test"]:
    # data/RepCount/annotation/train.csv for example
    df = pd.read_csv(f"{root}/annotation/{split}.csv")
    # L1, L2, L3, L4, ... are the frame numbers where reps start and end
    label_cols = [col for col in df.columns if col.startswith("L")]

    for _, row in df.iterrows():
        exercise = CLASS_NAMES.get(row["type"])
        if exercise is None:
            skipped.add(row["type"])
            continue
        if row["count"] == 0:
            continue

        # build path to the video file
        path = f"{root}/video/{split}/{row['name']}"

        if not os.path.exists(path):
            continue

        # Group the 'L's in pairs
        # A pair tells you the len in frames of a rep
        # A list of (start, end) frames per rep
        frames = [int(row[col]) for col in label_cols if pd.notna(row[col])]
        reps = [(frames[i], frames[i + 1]) for i in range(0, len(frames) - 1, 2)]

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # save the record
        videos.append({"split": split, "name": row["name"], "path": path, "class": exercise, "fps": fps, "reps": reps})

os.makedirs("figures", exist_ok=True)
json.dump(videos, open("data/metadata.json", "w"), indent=1)

# Count videos per class
counts = pd.DataFrame(videos).groupby(["class", "split"]).size().unstack(fill_value=0)
print(counts)
print(f"Total gym videos kept: {len(videos)}")

# basic plotting of class counts per exercise
counts.plot(kind="bar", figsize=(9, 5))
plt.ylabel("Number of videos")
plt.title("Videos per exercise (gym exercises in RepCount Part-A)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("figures/class_counts.png")

print("Skipped classes:", skipped)