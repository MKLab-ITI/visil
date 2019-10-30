from __future__ import unicode_literals
import cv2
import numpy as np


def resize_frame(frame, desired_size):
    min_size = np.min(frame.shape[:2])
    ratio = desired_size / min_size
    frame = cv2.resize(frame, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return frame


def load_video(video):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if isinstance(frame, np.ndarray):
            if int(count % round(fps)) == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(resize_frame(frame, 256))
        else:
            break
        count += 1
    cap.release()
    return np.array(frames)
