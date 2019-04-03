# Argument scopes for VGG models
import tensorflow as tf

slim = tf.contrib.slim

def vgg_arg_scope(weight_decay=0.0005, use_batch_norm=False, is_training=True, batch_norm_decay=0.95, use_scale=True):
  """Defines the VGG arg scope.

  Returns:
    An arg_scope.
  """
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params={'is_training': is_training, 'decay': batch_norm_decay, 'updates_collections': None, 'scale': use_scale}
  else:
    normalizer_fn = None
    normalizer_params = {}

  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                      biases_initializer=tf.constant_initializer(value=0.0, dtype=tf.float32), 
                      normalizer_fn=normalizer_fn, 
                      normalizer_params=normalizer_params):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      with slim.arg_scope([slim.batch_norm], **normalizer_params) as arg_sc:
        return arg_sc