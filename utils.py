import cv2
import numpy as np
import os


def center_crop(frame, desired_size):
    if frame.ndim == 3:
        old_size = frame.shape[:2]
        top = int(np.maximum(0, (old_size[0] - desired_size)/2))
        left = int(np.maximum(0, (old_size[1] - desired_size)/2))
        return frame[top: top+desired_size, left: left+desired_size, :]
    else: 
        old_size = frame.shape[1:3]
        top = int(np.maximum(0, (old_size[0] - desired_size)/2))
        left = int(np.maximum(0, (old_size[1] - desired_size)/2))
        return frame[:, top: top+desired_size, left: left+desired_size, :]


def resize_frame(frame, desired_size):
    min_size = np.min(frame.shape[:2])
    ratio = desired_size / min_size
    frame = cv2.resize(frame, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return frame

def load_video(video, all_frames=False, fps=1, cc_size=224, rs_size=256):
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(video)
    fps_div = fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 144 or fps is None:
        fps = 25
    frames = []
    count = 0
    while cap.isOpened():
        ret = cap.grab()
        if int(count % round(fps / fps_div)) == 0 or all_frames:
            ret, frame = cap.retrieve()
            if isinstance(frame, np.ndarray):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if rs_size is not None:
                    frame = resize_frame(frame, rs_size)
                frames.append(frame)
            else:
                break
        count += 1
    cap.release()
    frames = np.array(frames)
    if cc_size is not None:
        frames = center_crop(frames, cc_size)
    return frames

def load_video_2(folder_address, aweme_id, cc_size=224, rs_size=256):
    img_list = [] # ['12345678_0.jpg', '23456789_1.jpg']
    all_file = os.listdir(folder_address)
    for single_file in all_file:
        if str(aweme_id) in single_file:
            img_list.append(single_file)
    frames = []
    for img in img_list:
        img_address = os.path.join(folder_address, img)
        single_image = cv2.imread(img_address)
        if isinstance(single_image, np.ndarray):
            single_image = cv2.cvtColor(single_image, cv2.COLOR_BGR2RGB)
            if rs_size is not None:
                single_image = resize_frame(single_image, rs_size)
                frames.append(single_image)
    frames = np.array(frames)
    if cc_size is not None:
        frames = center_crop(frames, cc_size)
    return frames
