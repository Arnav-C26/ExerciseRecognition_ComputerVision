# Make the report figures from the saved embeddings.
# Usage
# RUN IN ROOT OF FOLDER
# `python src/plots.py tsne` -> t-SNE of videos colored by exercise
# `python src/plots.py ssm <video>` -> self-similarity matrix for one video

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

videos = json.load(open("data/metadata.json"))
videos = [video for video in videos if os.path.exists("data/embeddings/" + video["name"].replace(".mp4", ".npz"))]

def load(video):
    return np.load("data/embeddings/" + video["name"].replace(".mp4", ".npz"))

# visualize clusters
def tsne_plot():
    # average the frame embeddings to get one vector per video
    # dim = (number of videos, 384)
    video_vectors = np.stack([load(video)["emb"].mean(axis=0) for video in videos])
    labels = np.array([video["class"] for video in videos])

    # apply tsne
    # standard perplexity = 30
    points_2d = TSNE(perplexity=30, random_state=42).fit_transform(video_vectors)

    plt.figure(figsize=(7, 6))
    # labels == exercise gives true or false for each video
    # so points_2d[labels == exercise, 0] means
    # 'the x-coordinates of just this type of exercise' for example
    for exercise in sorted(set(labels)):
        plt.scatter(points_2d[labels == exercise, 0], points_2d[labels == exercise, 1], label=exercise)
    plt.legend()
    plt.title("t-SNE of DINOv2 video embeddings")
    plt.savefig("figures/tsne.png")

# for one video, compare every frame to each other frame and show results in heatmap
def ssm_plot(name):
    video = next(video for video in videos if video["name"] == name)
    data = load(video)
    embeddings = data["emb"] / np.linalg.norm(data["emb"], axis=1, keepdims=True)
    similarity = embeddings @ embeddings.T # the cosine similarity between every pair of frames

    plt.figure(figsize=(6, 6))
    plt.imshow(similarity)
    # mark where each labeled rep starts
    for start, end in video["reps"]:
        row = np.searchsorted(data["frame_idx"], start)
        plt.axvline(row, color="white", lw=0.5)
        plt.axhline(row, color="white", lw=0.5)
    plt.title(f"{name} ({video['class']}, {len(video['reps'])} reps)")
    plt.xlabel("frame")
    plt.ylabel("frame")
    plt.savefig("figures/ssm_" + name.replace(".mp4", ".png"))

if sys.argv[1] == "tsne":
    tsne_plot()
else:
    ssm_plot(sys.argv[2])