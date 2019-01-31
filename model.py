import tensorflow as tf
import os
import numpy as np

class neural_network(object):

    def __init__(self, sess, input_size, learning_rate=1e-4):

        self._sess = sess
        self._input_size = input_size
        self._lr = learning_rate

        self._build_graph()

    def _build_graph(self):
        with tf.variable_scope('input'):
            self._input = tf.placeholder(tf.float32, shape=[None, int(np.prod(self._input_size))])
            self._label = tf.placeholder(tf.float32, shape=[None, 1])

        with tf.variable_scope('classify'):
            self._result = tf.layers.dense(self._input, units=10, activation='relu')
            self._result = tf.layers.dense(self._result, units=1, activation='sigmoid')

        with tf.variable_scope('loss'):
            self._loss = self.train_loss(self._result, self._label)

            self._test_loss = self._loss

        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self._lr, beta1=0.5, beta2=0.9)

        with tf.variable_scope('training-step'):
            self._train = optimizer.minimize(self._loss)

        self.saver = tf.train.Saver(max_to_keep=None)
        init = tf.initializers.global_variables()
        self._sess.run(init)

    @staticmethod
    def train_loss(pred, ground_truth):
        loss = tf.keras.backend.binary_crossentropy(ground_truth, pred)
        return loss

    def update(self, X, Y):
        _, loss = self._sess.run([self._train, self._loss], feed_dict = {self._input : X, self._label: Y})
        return loss

    def save_model(self, index, outdir):
        save = self.saver.save(self._sess, os.path.join(outdir, 'model' , 'model_{}'.format(index)))
        return save

    def restore_model(self, path):
        self.saver.restore(self._sess, path)

    def test(self, X, Y):
        test_loss, predict = self._sess.run([self._test_loss, self._result], feed_dict = {self._input: X, self._label: Y})
        return test_loss, predict

