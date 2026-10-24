# Extract MediaPipe Pose keypoints for each video at 10 fps (same frame sampling as embed.py)
# Usage:
# RUN IN ROOT OF FOLDER
# `python src/keypoints.py data/sample/stu9_64.mp4` (one video)
# OR
# `python src/keypoints.py` (for all videos in data/metadata.json, takes a while)

import sys
import os
import json
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

TARGET_FPS = 10
NUM_LANDMARKS = 33  # MediaPipe for Pose Landmarking always outputs 33 body landmarks

# We download the pose landmarker model file once and store it in data/models
MODEL_PATH = "data/models/pose_landmarker_full.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")

def get_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok = True)
        print("Currently downloading the pose model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# We create a new landmarker for each video so that it resets between video files. Since we only care about
# the main subject, we set num_poses = 1.
def make_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options = mp_python.BaseOptions(model_asset_path = MODEL_PATH),
        running_mode = vision.RunningMode.VIDEO,
        num_poses = 1,
    )
    return vision.PoseLandmarker.create_from_options(options)

# This function reads in the video and keeps 1 out of every step frame. MediaPipe Pose Landmarker is
# then run on the kept frames. If a frame doesn't consist of anyone, it's filled with NaN.
def extract(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # We skip corrupted videos that report 0 fps
    if fps <= 0:
        cap.release()
        return None

    # Determine how many frames to skip (same formula as embed.py): frame rate / target fps, ex. 30 / 10 = 3, 
    # so we keep 1 frame out of every 3
    step = max(round(fps / TARGET_FPS), 1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    kp = []     # Image-based landmarks: x, y, z, visibility
    world = []  # Real-world landmarks in meters (centered on the hips)
    idx = []    # Original frame number of each kept frame
    with make_landmarker() as landmarker:
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                # The timestamp is in milliseconds (frame number / fps * 1000)
                result = landmarker.detect_for_video(image, int(i * 1000 / fps))

                if result.pose_landmarks:
                    kp.append([[l.x, l.y, l.z, l.visibility] for l in result.pose_landmarks[0]])
                    world.append([[l.x, l.y, l.z] for l in result.pose_world_landmarks[0]])
                else:
                    # Fill with NaN if no person was found
                    kp.append(np.full((NUM_LANDMARKS, 4), np.nan))
                    world.append(np.full((NUM_LANDMARKS, 3), np.nan))
                idx.append(i)
            i += 1
    cap.release()

    # If the video has no readable frames at all
    if len(idx) == 0:
        return None
    
    return (np.array(kp, dtype = np.float32), np.array(world, dtype = np.float32),
            np.array(idx), np.array([width, height]))

# This portion only runs when we run the file directly (python src/keypoints.py). If a video path was given on the 
# command line, we only process that one video. Otherweise, we process every video listed in metadata.json.
if __name__ == "__main__":
    if len(sys.argv) > 1:
        paths = [sys.argv[1]]
    else:
        paths = [video["path"] for video in json.load(open("data/metadata.json"))]

    get_model()
    os.makedirs("data/keypoints", exist_ok = True)
    for n, path in enumerate(paths):
        out = "data/keypoints/" + os.path.basename(path).replace(".mp4", ".npz")
        if os.path.exists(out):
            continue
        res = extract(path)
        if res is None:
            print(f"could not read {path}, skipping")
            continue
        kp, world, idx, size = res
        # Everything is saved into a .npz file
        np.savez(out, kp = kp, world = world, frame_idx = idx, size = size)
        found = 100 * np.mean(~np.isnan(kp[:, 0, 0]))
        print(f"[{n + 1}/{len(paths)}] {path} (pose found in {found:.0f}% of frames)")