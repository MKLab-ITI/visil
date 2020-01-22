import numpy as np
import tensorflow as tf

from .layers import PCA_layer, Attention_layer, Video_Comparator
from .similarity import chamfer_similarity, symmetric_chamfer_similarity


class ViSiL(object):

    def __init__(self, model_dir, net='resnet', load_queries=False,
                 dims=None, whitening=True, attention=True, video_comparator=True,
                 queries_number=None, gpu_id=0, similarity_function='chamfer'):

        self.net = net
        if self.net not in ['resnet', 'i3d']:
            raise Exception('[ERROR] Not supported backbone network: {}. '
                            'Supported options: resnet or i3d'.format(self.net))

        self.load_queries = load_queries
        if whitening or dims is not None:
            self.PCA = PCA_layer(dims=dims, whitening=whitening, net=self.net)
        if attention:
            if whitening and dims is None:
                self.att = Attention_layer(shape=400 if self.net == 'i3d' else 3840)
            else:
                print('[WARNING] Attention layer has been deactivated. '
                      'It works only with Whitening layer of {} dimensions. '.
                      format(400 if self.net == 'i3d' else 3840))
        if video_comparator:
            if whitening and attention and dims is None:
                self.vid_comp = Video_Comparator()
            else:
                print('[WARNING] Video comparator has been deactivated. '
                      'It works only with Whitening layer of {} dimensions '
                      'and Attention layer. '.format(400 if self.net == 'i3d' else 3840))

        if similarity_function == 'chamfer':
            self.f2f_sim = lambda x: chamfer_similarity(x, max_axis=2, mean_axis=1)
            self.v2v_sim = lambda x: chamfer_similarity(x, max_axis=1, mean_axis=0)
        elif similarity_function == 'symmetric_chamfer':
            self.f2f_sim = lambda x: symmetric_chamfer_similarity(x, axes=[1, 2])
            self.v2v_sim = lambda x: symmetric_chamfer_similarity(x, axes=[0, 1])
        else:
            raise Exception('[ERROR] Not implemented similarity function: {}. '
                            'Supported options: chamfer or symmetric_chamfer'.format(similarity_function))

        self.frames = tf.placeholder(tf.uint8, shape=(None, None, None, 3), name='input')
        with tf.device('/cpu:0'):
            if self.net == 'resnet':
                processed_frames = self.preprocess_resnet(self.frames)
            elif self.net == 'i3d':
                processed_frames = self.preprocess_i3d(self.frames)

        with tf.device('/gpu:%i' % gpu_id):
            self.region_vectors = self.extract_region_vectors(processed_frames)
            if self.load_queries:
                print('[INFO] Queries will be loaded to the gpu')
                self.queries = [tf.Variable(np.zeros((1, 9, 3840)), dtype=tf.float32,
                                            validate_shape=False) for _ in range(queries_number)]
                self.target = tf.placeholder(tf.float32, [None, None, None], name='target')
                self.similarities = []
                for q in self.queries:
                    sim_matrix = self.frame_to_frame_similarity(q, self.target)
                    similarity = self.video_to_video_similarity(sim_matrix)
                    self.similarities.append(similarity)
            else:
                print('[INFO] Queries will NOT be loaded to the gpu')
                self.query = tf.placeholder(tf.float32, [None, None, None], name='query')
                self.target = tf.placeholder(tf.float32, [None, None, None], name='target')
                sim_matrix = self.frame_to_frame_similarity(self.query, self.target)
                self.similarity = self.video_to_video_similarity(sim_matrix)

        init = self.load_model(model_dir)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(init)

    def preprocess_resnet(self, video):
        from .nets import vgg_preprocessing
        video = tf.map_fn(lambda x: vgg_preprocessing.preprocess_for_eval(x, 224, 224, 256), video,
                          parallel_iterations=100, dtype=tf.float32, swap_memory=True)
        return video

    def preprocess_i3d(self, video):
        video = tf.image.resize_with_crop_or_pad(video, 224, 224)
        video = tf.image.convert_image_dtype(video, dtype=tf.float32)
        video = tf.multiply(tf.subtract(video, 0.5), 2.0)
        video = tf.expand_dims(video, 0)
        return video

    def region_pooling(self, video):
        if self.net == 'resnet':
            from .nets import resnet_v1
            with tf.contrib.slim.arg_scope(resnet_v1.resnet_arg_scope()):
                _, network = resnet_v1.resnet_v1_50(video, num_classes=None, is_training=False)

            layers = [['resnet_v1_50/block1', 8], ['resnet_v1_50/block2', 4],
                      ['resnet_v1_50/block3', 2], ['resnet_v1_50/block4', 2]]

            with tf.variable_scope('region_vectors'):
                features = []
                for l, p in layers:
                    logits = tf.nn.relu(network[l])
                    logits = tf.layers.max_pooling2d(logits, [np.floor(p+p/2), np.floor(p+p/2)], p, padding='VALID')
                    logits = tf.nn.l2_normalize(logits, -1, epsilon=1e-15)
                    features.append(logits)
                logits = tf.concat(features, axis=-1)
                logits = tf.nn.l2_normalize(logits, -1, epsilon=1e-15)
            logits = tf.reshape(logits, [tf.shape(logits)[0], -1, tf.shape(logits)[-1]])
        elif self.net == 'i3d':
            from .nets import i3d
            with tf.variable_scope('RGB'):
                model = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint='Logits')
                logits, _ = model(video, is_training=False, dropout_keep_prob=1.0)

            with tf.variable_scope('region_vectors'):
                logits = tf.nn.l2_normalize(tf.nn.relu(logits), -1, epsilon=1e-15)

        return tf.expand_dims(logits, axis=1) if len(logits.shape) < 3 else logits

    def extract_region_vectors(self, video):
        logits = self.region_pooling(video)
        if hasattr(self, 'PCA'):
            logits = tf.nn.l2_normalize(self.PCA(logits), -1, epsilon=1e-15)
        if hasattr(self, 'att'):
            logits, weights = self.att(logits)
        return logits
    
    def frame_to_frame_similarity(self, query, target):
        tensor_dot = tf.tensordot(query, tf.transpose(target), axes=1)
        sim_matrix = self.f2f_sim(tensor_dot)
        return sim_matrix

    def video_to_video_similarity(self, sim):
        if hasattr(self, 'vid_comp'):
            sim = self.vid_comp(sim)
        sim = self.v2v_sim(sim)
        return sim

    def load_model(self, model_path):
        previous_variables = [var_name for var_name, _ in tf.contrib.framework.list_variables(model_path)]
        restore_map = {variable.op.name: variable for variable in tf.global_variables()
                       if variable.op.name in previous_variables and 'PCA' not in variable.op.name} #
        print('[INFO] {} layers loaded'.format(len(restore_map)))
        tf.contrib.framework.init_from_checkpoint(model_path, restore_map)
        tf_init = tf.global_variables_initializer()
        return tf_init

    def extract_features(self, frames, batch_sz):
        features = []
        for b in range(frames.shape[0] // batch_sz + 1):
            batch = frames[b * batch_sz: (b+1) * batch_sz]
            if batch.shape[0] > 0:
                if batch.shape[0] >= batch_sz or self.net == 'resnet':
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

    def add_query(self, query):
        if self.load_queries:
            raise Exception('[ERROR] Operation not permitted when queries are loaded to GPU.')
        else:
            self.queries.append(query)

    def calculate_similarities_to_queries(self, target):
        if self.load_queries:
            return self.sess.run(self.similarities, feed_dict={self.target: target})
        else:
            return [self.calculate_video_similarity(q, target) for q in self.queries]

    def calculate_video_similarity(self, query, target):
        return self.sess.run(self.similarity, feed_dict={self.query: query, self.target: target})
