import numpy as np
import tensorflow as tf

from garage.tf.core.cnn import cnn
from tests.fixtures import TfTestCase


class TestCNN(TfTestCase):
    def setUp(self):
        self.obs_input = np.ones((2, 5, 4, 3))
        input_shape = self.obs_input.shape[1:]  # height, width, channel
        self.hidden_nonlinearity = tf.nn.relu

        self._input_ph = tf.placeholder(
            tf.float32, shape=(None, ) + input_shape, name="input")

        self._output_shape = 2

        # Build the default cnn
        with tf.variable_scope("CNN"):
            self.cnn = cnn(
                input_var=self._input_ph,
                output_dim=self._output_shape,
                filter_dims=(3, 3, 3),
                num_filters=(32, 64, 128),
                stride=1,
                name="cnn1",
                hidden_nonlinearity=self.hidden_nonlinearity)

        super(TestCNN, self).setUp()

    def test_shape(self):
        result = self.sess.run(
            self.cnn, feed_dict={self._input_ph: self.obs_input})
        assert result.shape[1] == self._output_shape

    def test_output(self):
        for s in tf.global_variables():
            print(s)
        # with tf.variable_scope("CNN", reuse=True):
        #     h1_w = tf.get_variable("mlp1/hidden_0/kernel")
        #     h1_b = tf.get_variable("mlp1/hidden_0/bias")
        #     h2_w = tf.get_variable("mlp1/hidden_1/kernel")
        #     h2_b = tf.get_variable("mlp1/hidden_1/bias")
        #     out_w = tf.get_variable("mlp1/output/kernel")
        #     out_b = tf.get_variable("mlp1/output/bias")
