"""CNN with tensorflow as the only dependency."""

import tensorflow as tf


def cnn(input_var,
        output_dim,
        filter_dims,
        num_filters,
        stride,
        name,
        padding="SAME",
        max_pooling=False,
        pool_shape=(2, 2),
        hidden_nonlinearity=tf.nn.relu,
        hidden_w_init=tf.contrib.layers.xavier_initializer(),
        hidden_b_init=tf.zeros_initializer(),
        output_nonlinearity=None,
        output_w_init=tf.contrib.layers.xavier_initializer(),
        output_b_init=tf.zeros_initializer()):
    """
    CNN function. Based on 'NHWC' data format: [batch, height, width, channel].

    Args:
        input_var: Input tf.Tensor to the CNN.
        output_dim: Dimension of the network output.
        filter_dims: Dimension of the filters.
        num_filters: Number of filters.
        stride: The stride of the sliding window.
        name: Variable scope of the cnn.
        padding: The type of padding algorithm to use, from "SAME", "VALID".
        max_pooling: Boolean for using max pooling layer or not.
        pool_shape: Dimension of the pooling layer(s).
        hidden_nonlinearity: Activation function for
                    intermediate dense layer(s).
        hidden_w_init: Initializer function for the weight
                    of intermediate dense layer(s).
        hidden_b_init: Initializer function for the bias
                    of intermediate dense layer(s).
        output_nonlinearity: Activation function for
                    output dense layer.
        output_w_init: Initializer function for the weight
                    of output dense layer(s).
        output_b_init: Initializer function for the bias
                    of output dense layer(s).

    Return:
        The output tf.Tensor of the CNN.
    """
    strides = [1, stride, stride, 1]
    pool_shape = [1, pool_shape[0], pool_shape[1], 1]

    with tf.variable_scope(name):
        h = input_var
        for index, (filter_dim, num_filter) in enumerate(
                zip(filter_dims, num_filters)):
            h = hidden_nonlinearity(
                _conv(h, 'h{}'.format(index), filter_dim, num_filter, strides,
                      hidden_w_init, hidden_b_init, padding))
            if max_pooling:
                h = tf.nn.max_pool(
                    h, ksize=pool_shape, strides=strides, padding=padding)
        # convert conv to dense
        dim = tf.reduce_prod(h.get_shape()[1:].as_list())
        h = tf.reshape(h, [-1, dim.eval()])
        h = tf.layers.dense(
            inputs=h,
            units=output_dim,
            activation=output_nonlinearity,
            kernel_initializer=output_w_init,
            bias_initializer=output_b_init,
            name="output")

        return h


def _conv(input_var,
          name,
          filter_size,
          num_filter,
          strides,
          hidden_w_init,
          hidden_b_init,
          padding="SAME"):

    # channel from input
    input_shape = input_var.get_shape()[-1].value
    # [filter_height, filter_width, in_channels, out_channels]
    w_shape = [filter_size, filter_size, input_shape, num_filter]
    b_shape = [1, 1, 1, num_filter]

    with tf.variable_scope(name):
        w = tf.get_variable('w', w_shape, initializer=hidden_w_init)
        b = tf.get_variable('b', b_shape, initializer=hidden_b_init)

        return tf.nn.conv2d(input_var, w, strides=strides, padding=padding) + b
