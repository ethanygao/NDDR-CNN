import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


def xavier_init_numpy(fan_in, fan_out, shape):
    low = -np.sqrt(6.0 / (fan_in + fan_out))
    high = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(low=low, high=high, size=shape)


def nddr_layer(net_1, net_2, layer_ind=1, depth=None, scope=None, init_weights=[0.9, 0.1], init_method='constant'):
    if depth == None:
        depth = net_1.get_shape().as_list()[-1]

    if scope == None:
        scope = ('nddr_%d' % layer_ind)

    with tf.variable_scope(scope) as sc:
        end_points_collection = 'nddr_end_points'
        with slim.arg_scope([slim.conv2d], outputs_collections=end_points_collection):
            if init_method == 'constant':
                diag = np.concatenate(
                    (init_weights[0] * np.diag(np.ones(depth)), init_weights[1] * np.diag(np.ones(depth))),
                    axis=-2).astype(dtype=np.float32)
                diag = diag[np.newaxis, np.newaxis]
                initializer_conv_1 = tf.constant_initializer(diag)

                diag = np.concatenate(
                    (init_weights[1] * np.diag(np.ones(depth)), init_weights[0] * np.diag(np.ones(depth))),
                    axis=-2).astype(dtype=np.float32)
                diag = diag[np.newaxis, np.newaxis]
                initializer_conv_2 = tf.constant_initializer(diag)
            elif init_method == 'xavier':
                initializer_conv_1 = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
                initializer_conv_2 = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
            else:
                raise NotImplementedError

            net_nddr_concat = tf.concat([net_1, net_2], axis=-1, name=scope + '_concat')

            with slim.arg_scope([slim.conv2d], biases_initializer=tf.zeros_initializer()):
                net_1 = slim.conv2d(net_nddr_concat, depth, [1, 1], scope=scope + '_n1',
                                    weights_initializer=initializer_conv_1)
                net_2 = slim.conv2d(net_nddr_concat, depth, [1, 1], scope=scope + '_n2',
                                    weights_initializer=initializer_conv_2)

        return net_1, net_2