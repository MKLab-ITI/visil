import numpy as np
import tensorflow as tf


class PCA_layer(object):

    def __init__(self, whitening=True, dims=None, net='resnet'):
        pca = np.load('ckpt/{}/pca.npz'.format(net))
        with tf.variable_scope('PCA'):
            self.mean = tf.get_variable('mean_sift',
                                        initializer=pca['mean'],
                                        dtype=tf.float32,
                                        trainable=False)

            weights = pca['V'][:, :dims]
            if whitening:
                d = pca['d'][:dims]
                D = np.diag(1. / np.sqrt(d))
                weights = np.dot(D, weights.T).T

            self.weights = tf.get_variable('weights',
                                           initializer=weights,
                                           dtype=tf.float32,
                                           trainable=False)

    def __call__(self, logits):
        logits = logits - self.mean
        logits = tf.tensordot(logits, self.weights, axes=1)
        return logits


class Attention_layer(object):

    def __init__(self, shape=3840):
        with tf.variable_scope('attention_layer'):
            self.context_vector = tf.get_variable('context_vector', shape=(shape, 1),
                                                  dtype=tf.float32, trainable=False)

    def __call__(self, logits):
        weights = tf.tensordot(logits, self.context_vector, axes=1) / 2.0 + 0.5
        return tf.multiply(logits, weights), weights


class Video_Comparator(object):

    def __init__(self):
        self.conv1 = tf.keras.layers.Conv2D(32, [3, 3], activation='relu')
        self.mpool1 = tf.keras.layers.MaxPool2D([2, 2], 2)
        self.conv2 = tf.keras.layers.Conv2D(64, [3, 3], activation='relu')
        self.mpool2 = tf.keras.layers.MaxPool2D([2, 2], 2)
        self.conv3 = tf.keras.layers.Conv2D(128, [3, 3], activation='relu')
        self.fconv = tf.keras.layers.Conv2D(1, [1, 1])

    def __call__(self, sim_matrix):
        with tf.variable_scope('video_comparator'):
            sim = tf.reshape(sim_matrix, (1, tf.shape(sim_matrix)[0], tf.shape(sim_matrix)[1], 1))
            sim = tf.pad(sim, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
            sim = self.conv1(sim)
            sim = self.mpool1(sim)
            sim = tf.pad(sim, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
            sim = self.conv2(sim)
            sim = self.mpool2(sim)
            sim = tf.pad(sim, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
            sim = self.conv3(sim)
            sim = self.fconv(sim)
            sim = tf.clip_by_value(sim, -1.0, 1.0)
            sim = tf.squeeze(sim, [0, 3])
        return sim
