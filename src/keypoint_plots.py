# Make the Approach 1 report figures from the saved keypoints.
# Usage
# RUN IN ROOT OF FOLDER
# `python src/keypoint_plots.py skeleton <video>` (skeleton drawn over frames)
# `python src/keypoint_plots.py angles <video>` (angles of body joints over time with rep starts marked)
# `python src/keypoint_plots.py detection` (how often MediaPipe found a person per exercise)

import sys
import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from features import compute, ANGLES

videos = json.load(open("data/metadata.json"))
videos = [video for video in videos if os.path.exists("data/keypoints/" + video["name"].replace(".mp4", ".npz"))]

# We use Google's MediaPipe joint markings/configuration to determine the body parts we want to connect (to form a line)
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),          # shoulders and arms
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),  # hands
    (11, 23), (12, 24), (23, 24),                              # torso
    (23, 25), (25, 27), (24, 26), (26, 28),                    # legs
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),  # feet
]

def load(video):
    return np.load("data/keypoints/" + video["name"].replace(".mp4", ".npz"))

def get_video(name):
    return next(video for video in videos if video["name"] == name)

# In this function, we create a skeleton plot, which depicts the skeleton on a few frames (the start and the middle of the first 3 reps)
def skeleton_plot(name):
    video = get_video(name)
    data = load(video)
    # Set the path as the sample folder
    path = video["path"] if os.path.exists(video["path"]) else "data/sample/" + name
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Unable to open the video: {path}")

    shots = []
    for r, (start, end) in enumerate(video["reps"][:3]):
        shots.append((f"rep {r + 1} start", np.searchsorted(data["frame_idx"], start)))
        shots.append((f"rep {r + 1} middle", np.searchsorted(data["frame_idx"], (start + end) // 2)))
    fig, axes = plt.subplots(2, 3, figsize = (12, 8))
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")

    for ax, (title, row) in zip(axes, shots):
        row = min(row, len(data["frame_idx"]) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(data["frame_idx"][row]))
        ret, frame = cap.read()
        if not ret:
            continue
        height, width = frame.shape[:2]
        points = data["kp"][row, :, :2] * np.array([width, height])
        if np.isnan(points[0, 0]):
            title += " (no pose found)"
        else:
            for a, b in CONNECTIONS:
                cv2.line(frame, tuple(points[a].astype(int)), tuple(points[b].astype(int)), (0, 255, 0), 3)
            for x, y in points[11:]:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
    cap.release()

    plt.suptitle(f"{name} ({video['class']}, {len(video['reps'])} reps)")
    plt.tight_layout()
    plt.savefig("figures/skeleton_" + name.replace(".mp4", ".png"))

# This function creates an angles plot, depicting the angles of joints over time
def angles_plot(name):
    video = get_video(name)
    data = load(video)
    res = compute(data)
    angles = res["angles"]
    found = 100 * res["detected"].mean()
    names = list(ANGLES.keys())

    fig, axes = plt.subplots(4, 1, figsize = (10, 9), sharex = True)
    for ax, joint in zip(axes, ["elbow", "shoulder", "hip", "knee"]):
        ax.plot(angles[:, names.index("l_" + joint)], label = "left")
        ax.plot(angles[:, names.index("r_" + joint)], label = "right")
        for start, end in video["reps"]:
            ax.axvline(np.searchsorted(data["frame_idx"], start), color = "gray", lw = 0.5, ls = "--")
        ax.set_ylabel(f"{joint} angle (deg)")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("frame")
    axes[0].set_title(f"{name} ({video['class']}, {len(video['reps'])} reps, pose found in {found:.0f}% of frames)")
    plt.tight_layout()
    plt.savefig("figures/angles_" + name.replace(".mp4", ".png"))

# The detection plot function depicts how often MediaPipe detected the subject for each exercise
def detection_plot():
    rows = [{"class": video["class"], "split": video["split"],
             "rate": np.mean(~np.isnan(load(video)["kp"][:, 0, 0]))} for video in videos]
    rates = pd.DataFrame(rows).groupby("class")["rate"].mean()
    print(rates)

    rates.plot(kind = "bar", figsize = (9, 5))
    plt.ylim(0, 1)
    plt.ylabel("Fraction of frames with a pose found")
    plt.title("MediaPipe Pose detection rate per exercise")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("figures/detection_rate.png")

if __name__ == "__main__":
    os.makedirs("figures", exist_ok = True)
    if sys.argv[1] == "skeleton":
        skeleton_plot(sys.argv[2])
    elif sys.argv[1] == "angles":
        angles_plot(sys.argv[2])
    else:
        detection_plot()


