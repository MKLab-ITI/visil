import os
import numpy as np
import tensorflow as tf

from utils import load_video


class VideoGenerator(tf.keras.utils.Sequence):
    def __init__(self, video_file):
        super(VideoGenerator, self).__init__()
        self.videos = np.loadtxt(video_file, dtype=str)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        return load_video(self.videos[index][1]), self.videos[index][0]
