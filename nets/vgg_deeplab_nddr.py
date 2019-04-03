from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from net.nddr_layer import nddr_layer

slim = tf.contrib.slim


def vgg_16_deeplab_nddr(inputs,
                        num_classes_1=21,
                        num_classes_2=21,
                        is_training=True,
                        dropout_keep_prob=0.5,
                        scope='vgg_16',
                        init_method='constant',
                        init_weights=[0.9, 0.1]):
    """NDDR VGG-16 Deeplab lfov model.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      scope: Optional scope for the variables.
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes_1: number of predicted classes for task 1.
      num_classes_2: number of predicted classes for task 2.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      scope: Optional scope for the variables.
      init_method: initializing method for weights of NDDR layers. {'constant, xavier'}
      init_weights: initializing weights for NDDR layers if init_method is 'constant'.


    Returns:
      the last ops for the two tasks, the end_points dicts for the two tasks, the end_points for the NDDR layers.
    """
    with tf.variable_scope(scope + '_1_1', 'vgg_16_1', [inputs]) as sc:
        end_points_collection_1 = 'vgg_16_1_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_1):
            net_1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net_1 = slim.max_pool2d(net_1, [3, 3], stride=2, padding='SAME', scope='pool1')

    with tf.variable_scope(scope + '_2_1', 'vgg_16_2', [inputs]) as sc:
        end_points_collection_2 = 'vgg_16_2_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_2):
            net_2 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net_2 = slim.max_pool2d(net_2, [3, 3], stride=2, padding='SAME', scope='pool1')

    layer_ind = 0
    layer_ind += 1
    net_1, net_2 = nddr_layer(net_1, net_2, layer_ind, init_weights=init_weights, init_method=init_method)

    with tf.variable_scope(scope + '_1_2', 'vgg_16_1') as sc:
        end_points_collection_1 = 'vgg_16_1_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_1):
            net_1 = slim.repeat(net_1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net_1 = slim.max_pool2d(net_1, [3, 3], stride=2, padding='SAME', scope='pool2')

    with tf.variable_scope(scope + '_2_2', 'vgg_16_2') as sc:
        end_points_collection_2 = 'vgg_16_2_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_2):
            net_2 = slim.repeat(net_2, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net_2 = slim.max_pool2d(net_2, [3, 3], stride=2, padding='SAME', scope='pool2')

    layer_ind += 1
    net_1, net_2 = nddr_layer(net_1, net_2, layer_ind, init_weights=init_weights, init_method=init_method)

    with tf.variable_scope(scope + '_1_3', 'vgg_16_1') as sc:
        end_points_collection_1 = 'vgg_16_1_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_1):
            net_1 = slim.repeat(net_1, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net_1 = slim.max_pool2d(net_1, [3, 3], stride=2, padding='SAME', scope='pool3')

    with tf.variable_scope(scope + '_2_3', 'vgg_16_2') as sc:
        end_points_collection_2 = 'vgg_16_2_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_2):
            net_2 = slim.repeat(net_2, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net_2 = slim.max_pool2d(net_2, [3, 3], stride=2, padding='SAME', scope='pool3')

    layer_ind += 1
    net_1, net_2 = nddr_layer(net_1, net_2, layer_ind, init_weights=init_weights, init_method=init_method)

    with tf.variable_scope(scope + '_1_4', 'vgg_16_1') as sc:
        end_points_collection_1 = 'vgg_16_1_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_1):
            net_1 = slim.repeat(net_1, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net_1 = slim.max_pool2d(net_1, [3, 3], stride=1, padding='SAME', scope='pool4')

    with tf.variable_scope(scope + '_2_4', 'vgg_16_2') as sc:
        end_points_collection_2 = 'vgg_16_2_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_2):
            net_2 = slim.repeat(net_2, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net_2 = slim.max_pool2d(net_2, [3, 3], stride=1, padding='SAME', scope='pool4')

    layer_ind += 1
    net_1, net_2 = nddr_layer(net_1, net_2, layer_ind, init_weights=init_weights, init_method=init_method)

    with tf.variable_scope(scope + '_1_5', 'vgg_16_1') as sc:
        end_points_collection_1 = 'vgg_16_1_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_1):
            net_1 = slim.repeat(net_1, 3, slim.conv2d, 512, [3, 3], rate=2, scope='conv5')
            net_1 = slim.max_pool2d(net_1, [3, 3], stride=1, padding='SAME', scope='pool5_max')
            net_1 = slim.avg_pool2d(net_1, [3, 3], stride=1, padding='SAME', scope='pool5_avg')

    with tf.variable_scope(scope + '_2_5', 'vgg_16_2') as sc:
        end_points_collection_2 = 'vgg_16_2_end_points_2'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_2):
            net_2 = slim.repeat(net_2, 3, slim.conv2d, 512, [3, 3], rate=2, scope='conv5')
            net_2 = slim.max_pool2d(net_2, [3, 3], stride=1, padding='SAME', scope='pool5_max')
            net_2 = slim.avg_pool2d(net_2, [3, 3], stride=1, padding='SAME', scope='pool5_avg')

    layer_ind += 1
    net_1, net_2 = nddr_layer(net_1, net_2, layer_ind, init_weights=init_weights, init_method=init_method)

    nddr_end_points = slim.utils.convert_collection_to_dict('nddr_end_points')

    with tf.variable_scope(scope + '_1_6', 'vgg_16_1') as sc:
        end_points_collection_1 = 'vgg_16_1_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_1):
            # Use conv2d instead of fully_connected layers.
            rate = 12
            net_1 = slim.conv2d(net_1, 1024, [3, 3], rate=rate, padding='SAME', scope='fc6')
            net_1 = slim.dropout(net_1, dropout_keep_prob, is_training=is_training,
                                 scope='dropout6')
            net_1 = slim.conv2d(net_1, 1024, [1, 1], padding='SAME', scope='fc7')
            net_1 = slim.dropout(net_1, dropout_keep_prob, is_training=is_training,
                                 scope='dropout7')
            net_1 = slim.conv2d(net_1, num_classes_1, [1, 1],
                                activation_fn=None,
                                normalizer_fn=None,
                                scope='fc8_voc12')
            end_points_1 = slim.utils.convert_collection_to_dict(end_points_collection_1)

    with tf.variable_scope(scope + '_2_6', 'vgg_16_2') as sc:
        end_points_collection_2 = 'vgg_16_2_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_2):
            # Use conv2d instead of fully_connected layers.
            rate = 12
            net_2 = slim.conv2d(net_2, 1024, [3, 3], rate=rate, padding='SAME', scope='fc6')
            net_2 = slim.dropout(net_2, dropout_keep_prob, is_training=is_training,
                                 scope='dropout6')
            net_2 = slim.conv2d(net_2, 1024, [1, 1], padding='SAME', scope='fc7')
            net_2 = slim.dropout(net_2, dropout_keep_prob, is_training=is_training,
                                 scope='dropout7')
            net_2 = slim.conv2d(net_2, num_classes_2, [1, 1],
                                activation_fn=None,
                                normalizer_fn=None,
                                scope='fc8_voc12')
            end_points_2 = slim.utils.convert_collection_to_dict(end_points_collection_2)

        return net_1, net_2, end_points_1, end_points_2, nddr_end_points


def vgg_16_shortcut_deeplab_nddr(inputs,
                                 num_classes_1=21,
                                 num_classes_2=21,
                                 is_training=True,
                                 dropout_keep_prob=0.5,
                                 scope='vgg_16',
                                 init_method='constant',
                                 init_weights=[0.9, 0.1]):
    """NDDR VGG-16 Deeplab lfov model with shortcut connections.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes_1: number of predicted classes for task 1.
      num_classes_2: number of predicted classes for task 2.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      scope: Optional scope for the variables.
      init_method: initializing method for weights of NDDR layers. {'constant, xavier'}
      init_weights: initializing weights for NDDR layers if init_method is 'constant'.


    Returns:
      the last ops for the two tasks, the end_points dicts for the two tasks, the end_points for the NDDR layers.
    """
    with tf.variable_scope(scope + '_1_1', 'vgg_16_1', [inputs]) as sc:
        end_points_collection_1 = 'vgg_16_1_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_1):
            net_1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net_1 = slim.max_pool2d(net_1, [3, 3], stride=2, padding='SAME', scope='pool1')

    with tf.variable_scope(scope + '_2_1', 'vgg_16_2', [inputs]) as sc:
        end_points_collection_2 = 'vgg_16_2_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_2):
            net_2 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net_2 = slim.max_pool2d(net_2, [3, 3], stride=2, padding='SAME', scope='pool1')

    layer_ind = 0
    layer_ind += 1
    net_1, net_2 = nddr_layer(net_1, net_2, layer_ind, init_weights=init_weights, init_method=init_method)
    nddr_1_1 = net_1
    nddr_1_2 = net_2

    with tf.variable_scope(scope + '_1_2', 'vgg_16_1') as sc:
        end_points_collection_1 = 'vgg_16_1_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_1):
            net_1 = slim.repeat(net_1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net_1 = slim.max_pool2d(net_1, [3, 3], stride=2, padding='SAME', scope='pool2')

    with tf.variable_scope(scope + '_2_2', 'vgg_16_2') as sc:
        end_points_collection_2 = 'vgg_16_2_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_2):
            net_2 = slim.repeat(net_2, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net_2 = slim.max_pool2d(net_2, [3, 3], stride=2, padding='SAME', scope='pool2')

    layer_ind += 1
    net_1, net_2 = nddr_layer(net_1, net_2, layer_ind, init_weights=init_weights, init_method=init_method)
    nddr_2_1 = net_1
    nddr_2_2 = net_2

    with tf.variable_scope(scope + '_1_3', 'vgg_16_1') as sc:
        end_points_collection_1 = 'vgg_16_1_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_1):
            net_1 = slim.repeat(net_1, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net_1 = slim.max_pool2d(net_1, [3, 3], stride=2, padding='SAME', scope='pool3')

    with tf.variable_scope(scope + '_2_3', 'vgg_16_2') as sc:
        end_points_collection_2 = 'vgg_16_2_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_2):
            net_2 = slim.repeat(net_2, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net_2 = slim.max_pool2d(net_2, [3, 3], stride=2, padding='SAME', scope='pool3')

    layer_ind += 1
    net_1, net_2 = nddr_layer(net_1, net_2, layer_ind, init_weights=init_weights, init_method=init_method)
    nddr_3_1 = net_1
    nddr_3_2 = net_2

    with tf.variable_scope(scope + '_1_4', 'vgg_16_1') as sc:
        end_points_collection_1 = 'vgg_16_1_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_1):
            net_1 = slim.repeat(net_1, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net_1 = slim.max_pool2d(net_1, [3, 3], stride=1, padding='SAME', scope='pool4')

    with tf.variable_scope(scope + '_2_4', 'vgg_16_2') as sc:
        end_points_collection_2 = 'vgg_16_2_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_2):
            net_2 = slim.repeat(net_2, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net_2 = slim.max_pool2d(net_2, [3, 3], stride=1, padding='SAME', scope='pool4')

    layer_ind += 1
    net_1, net_2 = nddr_layer(net_1, net_2, layer_ind, init_weights=init_weights, init_method=init_method)
    nddr_4_1 = net_1
    nddr_4_2 = net_2

    with tf.variable_scope(scope + '_1_5', 'vgg_16_1') as sc:
        end_points_collection_1 = 'vgg_16_1_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_1):
            net_1 = slim.repeat(net_1, 3, slim.conv2d, 512, [3, 3], rate=2, scope='conv5')
            net_1 = slim.max_pool2d(net_1, [3, 3], stride=1, padding='SAME', scope='pool5_max')
            net_1 = slim.avg_pool2d(net_1, [3, 3], stride=1, padding='SAME', scope='pool5_avg')

    with tf.variable_scope(scope + '_2_5', 'vgg_16_2') as sc:
        end_points_collection_2 = 'vgg_16_2_end_points_2'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_2):
            net_2 = slim.repeat(net_2, 3, slim.conv2d, 512, [3, 3], rate=2, scope='conv5')
            net_2 = slim.max_pool2d(net_2, [3, 3], stride=1, padding='SAME', scope='pool5_max')
            net_2 = slim.avg_pool2d(net_2, [3, 3], stride=1, padding='SAME', scope='pool5_avg')

    layer_ind += 1
    net_1, net_2 = nddr_layer(net_1, net_2, layer_ind, init_weights=init_weights, init_method=init_method)
    nddr_5_1 = net_1
    nddr_5_2 = net_2

    h = tf.shape(nddr_5_1)[1]
    w = tf.shape(nddr_5_1)[2]
    nddr_1_1_resize = tf.image.resize_bilinear(nddr_1_1, [h, w])
    nddr_1_2_resize = tf.image.resize_bilinear(nddr_1_2, [h, w])
    nddr_2_1_resize = tf.image.resize_bilinear(nddr_2_1, [h, w])
    nddr_2_2_resize = tf.image.resize_bilinear(nddr_2_2, [h, w])
    nddr_3_1_resize = tf.image.resize_bilinear(nddr_3_1, [h, w])
    nddr_3_2_resize = tf.image.resize_bilinear(nddr_3_2, [h, w])
    nddr_4_1_resize = tf.image.resize_bilinear(nddr_4_1, [h, w])
    nddr_4_2_resize = tf.image.resize_bilinear(nddr_4_2, [h, w])

    with slim.arg_scope([slim.conv2d], biases_initializer=tf.zeros_initializer(),
                        outputs_collections='nddr_end_points'):
        conv_concat_1 = tf.concat([nddr_1_1_resize, nddr_2_1_resize, nddr_3_1_resize, nddr_4_1_resize, nddr_5_1],
                                  axis=-1, name='conv_concat_1')
        conv_concat_2 = tf.concat([nddr_1_2_resize, nddr_2_2_resize, nddr_3_2_resize, nddr_4_2_resize, nddr_5_2],
                                  axis=-1, name='conv_concat_2')

        d = nddr_5_1.get_shape().as_list()[-1]
        net_1 = slim.conv2d(conv_concat_1, d, [1, 1], scope='nddr_shortcut_1')
        net_2 = slim.conv2d(conv_concat_2, d, [1, 1], scope='nddr_shortcut_2')

    nddr_end_points = slim.utils.convert_collection_to_dict('nddr_end_points')

    with tf.variable_scope(scope + '_1_6', 'vgg_16_1') as sc:
        end_points_collection_1 = 'vgg_16_1_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_1):
            # Use conv2d instead of fully_connected layers.
            rate = 12
            net_1 = slim.conv2d(net_1, 1024, [3, 3], rate=rate, padding='SAME', scope='fc6')
            net_1 = slim.dropout(net_1, dropout_keep_prob, is_training=is_training,
                                 scope='dropout6')
            net_1 = slim.conv2d(net_1, 1024, [1, 1], padding='SAME', scope='fc7')
            net_1 = slim.dropout(net_1, dropout_keep_prob, is_training=is_training,
                                 scope='dropout7')
            net_1 = slim.conv2d(net_1, num_classes_1, [1, 1],
                                activation_fn=None,
                                normalizer_fn=None,
                                scope='fc8_voc12')
            end_points_1 = slim.utils.convert_collection_to_dict(end_points_collection_1)

    with tf.variable_scope(scope + '_2_6', 'vgg_16_2') as sc:
        end_points_collection_2 = 'vgg_16_2_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection_2):
            # Use conv2d instead of fully_connected layers.
            rate = 12
            net_2 = slim.conv2d(net_2, 1024, [3, 3], rate=rate, padding='SAME', scope='fc6')
            net_2 = slim.dropout(net_2, dropout_keep_prob, is_training=is_training,
                                 scope='dropout6')
            net_2 = slim.conv2d(net_2, 1024, [1, 1], padding='SAME', scope='fc7')
            net_2 = slim.dropout(net_2, dropout_keep_prob, is_training=is_training,
                                 scope='dropout7')
            net_2 = slim.conv2d(net_2, num_classes_2, [1, 1],
                                activation_fn=None,
                                normalizer_fn=None,
                                scope='fc8_voc12')
            end_points_2 = slim.utils.convert_collection_to_dict(end_points_collection_2)

        return net_1, net_2, end_points_1, end_points_2, nddr_end_points
