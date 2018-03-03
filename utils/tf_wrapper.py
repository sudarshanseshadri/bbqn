import tensorflow as tf
from yang_pixelcnn.models import PixelCNN
from yang_pixelcnn.layers import *


class DotDict(object):
    def __init__(self, dict):
        self.dict = dict

    def __getattr__(self, name):
        return self.dict[name]

    def update(self, name, val):
        self.dict[name] = val

    # can delete this later
    def get(self, name):
        return self.dict[name]


class GatedPixelCNNWrapper(object):
    def __init__(self, FLAGS):
        # init model
        self.X = tf.placeholder(
            tf.float32,
            shape=[None, FLAGS.img_height, FLAGS.img_width, FLAGS.channel])
        self.model = PixelCNN(self.X, FLAGS)
        self.flags = FLAGS
        # init optimizer
        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=1e-3,decay=0.95,momentum=0.9).minimize(self.model.loss)
        # make tensorflow only use a limited amount of GPU memory
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train_batch(self, img_batch):
        batch_X = img_batch
        _, nll = self.sess.run(
            [self.optimizer, self.model.nll],
            feed_dict={self.X: batch_X})
        return nll


# specify command line arguments using flags
FLAGS = DotDict({
    'img_height': 42,
    'img_width': 42,
    'channel': 1,
    'data': 'mnist',
    'conditional': False,
    'num_classes': None,
    'filter_size': 3,
    'f_map': 32,
    'f_map_fc': 32,
    'colors': 8,
    'parallel_workers': 1,
    'layers': 21,
    'epochs': 25,
    'batch_size': 16,
    'model': '',
    'data_path': 'data',
    'ckpt_path': 'ckpts',
    'samples_path': 'samples',
    'summary_path': 'logs',
    'restore': True,
})