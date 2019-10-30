import numpy as np
import tensorflow as tf

from nets import resnet_v1, vgg_preprocessing


class PCA_layer(object):

    def __init__(self, whitening=True, dims=None):
        pca = np.load('model/pca.npz')
        with tf.variable_scope('PCA'):
            self.mean = tf.get_variable('mean_sift',
                                        initializer=pca['mean'],
                                        dtype=tf.float32,
                                        trainable=False)

            weights = pca['V'][:, :dims] if dims else pca['V']
            if whitening:
                d = pca['d'][:dims] if dims else pca['d']
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

    def __init__(self):
        with tf.variable_scope('attention_layer'):
            self.context_vector = tf.get_variable('context_vector', dtype=tf.float32,
                                                  trainable=False, shape=(3840, 1))

    def __call__(self, logits):
        weights = tf.tensordot(logits, self.context_vector, axes=1) / 2.0 + 0.5
        return tf.multiply(logits, weights), weights


class ViSiL(object):

    def __init__(self, model_dir, load_queries=False, queries_number=None, gpu_id=0):
        
        self.load_queries = load_queries
        self.frames = tf.placeholder(tf.uint8, shape=(None, None, None, 3), name='input')
        processed_frames = self.preprocess_video(self.frames)
        
        with tf.device('/gpu:%i' % gpu_id):
            with tf.contrib.slim.arg_scope(resnet_v1.resnet_arg_scope()):
                _, network = resnet_v1.resnet_v1_50(processed_frames, num_classes=None, is_training=False)
            self.region_vectors = self.extract_region_vectors(network)

            if self.load_queries:
                print('Queries will be loaded to the gpu')
                self.queries = [tf.Variable(np.zeros((1, 9, 3840)), dtype=tf.float32, validate_shape=False) for _ in range(queries_number)]
                self.candidate = tf.placeholder(tf.float32, [None, None, None], name='candidate')
                self.similarities = []
                for q in self.queries:
                    sim_matrix = self.frame_to_frame_similarity(q, self.candidate)
                    sim_matrix = self.video_to_video_similarity(sim_matrix)
                    sim_matrix = tf.squeeze(sim_matrix, [0, 3])
                    self.similarities.append(self.chamfer_similarity(sim_matrix))
            else:
                self.query = tf.placeholder(tf.float32, [None, None, None], name='query')
                self.candidate = tf.Variable(np.zeros((1, 9, 3840)), dtype=tf.float32, validate_shape=False)
                sim_matrix = self.frame_to_frame_similarity(self.query, self.candidate)
                sim_matrix = self.video_to_video_similarity(sim_matrix)
                sim_matrix = tf.squeeze(sim_matrix, [0, 3])
                self.similarity = self.chamfer_similarity(sim_matrix)
        
        init = self.load_model(model_dir)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(init)

    def preprocess_video(self, video):
        video = tf.map_fn(lambda x: vgg_preprocessing.preprocess_for_eval(x, 224, 224, 256), video,
                          parallel_iterations=100, dtype=tf.float32, swap_memory=True)
        return video

    def region_pooling(self, network):
        layers = [['resnet_v1_50/block1', 8], ['resnet_v1_50/block2', 4],
                  ['resnet_v1_50/block3', 2], ['resnet_v1_50/block4', 2]]
        
        with tf.variable_scope('region_vectors'):
            features = []
            for i, (l, p) in enumerate(layers):
                logits = tf.nn.relu(network[l])
                logits = tf.layers.max_pooling2d(logits, [np.ceil(p+p*0.4), np.ceil(p+p*0.4)], p, padding='VALID')
                logits = tf.nn.l2_normalize(logits, 3, epsilon=1e-15)
                features.append(logits)
            logits = tf.concat(features, axis=3)
            return tf.nn.l2_normalize(logits, 3, epsilon=1e-15)

    def extract_region_vectors(self, network):
        logits = self.region_pooling(network)
        logits = PCA_layer(whitening=True)(logits)
        logits = tf.nn.l2_normalize(logits, 3, epsilon=1e-15)
        logits, weights = Attention_layer()(logits)
        return tf.reshape(logits, [-1, tf.shape(logits)[1] * tf.shape(logits)[2], logits.shape[3]])

    def chamfer_similarity(self, sim, max_axis=1, mean_axis=0):
        sim = tf.reduce_max(sim, axis=max_axis)
        sim = tf.reduce_mean(sim, axis=mean_axis)
        return sim

    def frame_to_frame_similarity(self, query, candidate):
        tensor_dot = tf.tensordot(query, tf.transpose(candidate), axes=1)
        sim_matrix = self.chamfer_similarity(tensor_dot, max_axis=2, mean_axis=1)
        return sim_matrix

    def video_to_video_similarity(self, sim_matrix):
        sim = tf.reshape(sim_matrix, (1, tf.shape(sim_matrix)[0], tf.shape(sim_matrix)[1], 1))
        with tf.variable_scope('video_comparator'):
            sim = tf.pad(sim, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
            sim = tf.contrib.layers.conv2d(sim, 32, [3, 3], padding='VALID', activation_fn=tf.nn.relu,
                                           reuse=tf.AUTO_REUSE, scope='conv_0')
            sim = tf.layers.max_pooling2d(sim, [2, 2], 2, padding='VALID')

            sim = tf.pad(sim, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
            sim = tf.contrib.layers.conv2d(sim, 64, [3, 3], padding='VALID', activation_fn=tf.nn.relu,
                                           reuse=tf.AUTO_REUSE, scope='conv_1')
            sim = tf.layers.max_pooling2d(sim, [2, 2], 2, padding='VALID')

            sim = tf.pad(sim, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
            sim = tf.contrib.layers.conv2d(sim, 128, [3, 3], padding='VALID', activation_fn=tf.nn.relu,
                                           reuse=tf.AUTO_REUSE, scope='conv_2')

            sim = tf.contrib.layers.conv2d(sim, 1, [1, 1], padding='VALID', activation_fn=None,
                                           reuse=tf.AUTO_REUSE, scope='final_conv')
            sim = tf.clip_by_value(sim, -1.0, 1.0)
        return sim

    def load_model(self, model_path):
        previous_variables = [var_name for var_name, _ in tf.contrib.framework.list_variables(model_path)]
        restore_map = {variable.op.name: variable for variable in tf.global_variables()
                       if variable.op.name in previous_variables}
        print('{} layers loaded'.format(len(restore_map)))
        tf.contrib.framework.init_from_checkpoint(model_path, restore_map)
        tf_init = tf.global_variables_initializer()
        return tf_init

    def extract_features(self, frames, batch_sz):
        features = []
        for b in range(frames.shape[0] // batch_sz + 1):
            batch = frames[b * batch_sz: (b+1) * batch_sz]
            if batch.shape[0] > 0:
                features.append(self.sess.run(self.region_vectors, feed_dict={self.frames: batch}))
        features = np.concatenate(features, axis=0)
        while features.shape[0] < 8:
            features = np.concatenate([features, features], axis=0)
        return features

    def set_queries(self, queries):
        if self.load_queries:
            for i in range(len(queries)):
                self.sess.run(tf.assign(self.queries[i], queries[i], validate_shape=False))
        else:
            self.queries = queries

    def calculate_similarities(self, candidate_frames, batch_sz):
        candidate = self.extract_features(candidate_frames, batch_sz)
        if self.load_queries:
            return self.sess.run(self.similarities, feed_dict={self.candidate: candidate})
        else:
            self.sess.run(tf.assign(self.candidate, candidate, validate_shape=False))
            return [self.sess.run(self.similarity, feed_dict={self.query: q, self.candidate: candidate})
                    for q in self.queries]
