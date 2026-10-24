# Turn raw MediaPipe keypoints into features (per frame) for the RNN. We fill the frames with missing poses, then
# center on hips and scale by torso length, followed by the joint angles.
# Usage:
# RUN IN ROOT OF FOLDER
# `python src/features.py data/sample/stu9_64.mp4` (one video)
# OR
# `python src/features.py` (for all videos in data/metadata.json)
# Needs data/keypoints/<video>.npz from keypoints.py first.

import sys
import os
import json
import numpy as np

# MediaPipe Pose landmark numbers, defined by Google
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# Each angle is measured at the middle joint: (point a, vertex b, point c)
ANGLES = {
    "l_elbow": (L_SHOULDER, L_ELBOW, L_WRIST),
    "r_elbow": (R_SHOULDER, R_ELBOW, R_WRIST),
    "l_shoulder": (L_HIP, L_SHOULDER, L_ELBOW),
    "r_shoulder": (R_HIP, R_SHOULDER, R_ELBOW),
    "l_hip": (L_SHOULDER, L_HIP, L_KNEE),
    "r_hip": (R_SHOULDER, R_HIP, R_KNEE),
    "l_knee": (L_HIP, L_KNEE, L_ANKLE),
    "r_knee": (R_HIP, R_KNEE, R_ANKLE),
}

# Frames where no person was found are NaN
def fill_missing(x, detected):
    flat = x.reshape(len(x), -1).copy()
    t = np.arange(len(x))
    for d in range(flat.shape[1]):
        flat[:, d] = np.interp(t, t[detected], flat[detected, d])
    return flat.reshape(x.shape)

# We now make the keypoints independent of where the subject is in the frame, along with the camera's zoom. 
# We set the hips as (0, 0) so we can subtract from the midpoint of the hips. To elimiate zoom, we multiple the
# x and y by width/height, subtract the midpoint of the hips, and divide by the video's median torso length. 

def normalize_xy(kp, width, height):
    xy = kp[:, :, :2] * np.array([width, height])
    hips = xy[:, [L_HIP, R_HIP]].mean(axis = 1)
    shoulders = xy[:, [L_SHOULDER, R_SHOULDER]].mean(axis = 1)
    torso = np.median(np.linalg.norm(shoulders - hips, axis = 1))
    return (xy - hips[:, None]) / max(torso, 1e-6)

# We calculate the angle at point b between the lines b to a and b to c for every frame
def angle(a, b, c):
    ba, bc = a - b, c - b
    cos = (ba * bc).sum(axis = 1) / (np.linalg.norm(ba, axis = 1) * np.linalg.norm(bc, axis = 1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def compute(data):
    kp, world = data["kp"], data["world"]
    detected = ~np.isnan(kp[:, 0, 0])
    if detected.sum() == 0:
        return None
    kp = fill_missing(kp, detected)
    world = fill_missing(world, detected)

    width, height = data["size"]
    xy = normalize_xy(kp, width, height)
    angles = np.stack([angle(world[:, a], world[:, b], world[:, c]) for a, b, c in ANGLES.values()], axis = 1)
    return {"xy": xy, "angles": angles, "detected": detected}

# This portion only runs when we run the file directly (python src/features.py).
if __name__ == "__main__":
    if len(sys.argv) > 1:
        paths = [sys.argv[1]]
    else:
        paths = [video["path"] for video in json.load(open("data/metadata.json"))]

    os.makedirs("data/features", exist_ok = True)
    for n, path in enumerate(paths):
        name = os.path.basename(path).replace(".mp4", ".npz")
        src, out = "data/keypoints/" + name, "data/features/" + name
        if os.path.exists(out) or not os.path.exists(src):
            continue
        data = np.load(src)
        res = compute(data)
        if res is None:
            print(f"no pose found in {path}, skipping")
            continue
        feat = np.concatenate([res["xy"].reshape(len(res["xy"]), -1), res["angles"] / 180], axis = 1)
        np.savez(out, feat = feat.astype(np.float32), angles = res["angles"], detected = res["detected"],
                 frame_idx = data["frame_idx"])
        print(f"[{n + 1}/{len(paths)}] {path} -> {feat.shape}")
