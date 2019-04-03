import tensorflow as tf


def get_normal_loss(prediction, gt, num_classes, ignore_label):
    '''Compute normal loss. (normalized cosine distance)

    Args:
      prediction: the output of cnn.
      gt: the groundtruth.
      args: arguments.
    '''
    prediction = tf.reshape(prediction, [-1, num_classes])
    gt = tf.reshape(gt, [-1, num_classes])

    mask = tf.not_equal(gt, ignore_label)
    mask = tf.cast(tf.expand_dims(tf.reduce_any(mask, axis=-1), axis=-1), tf.float32)

    prediction = tf.nn.l2_normalize(prediction, dim=-1)
    gt = tf.nn.l2_normalize(tf.cast(gt, tf.float32), dim=-1)

    loss = tf.losses.cosine_distance(gt, prediction, dim=-1, weights=mask)

    return loss


def get_seg_loss(prediction, gt, num_classes, ignore_label):
    '''Compute seg loss. (softmax cross entropy)

    Args:
      prediction: the output of cnn.
      gt: the groundtruth.
      args: arguments.
    '''
    prediction = tf.reshape(prediction, [-1, num_classes])
    gt = tf.reshape(gt, [-1, ])
    indices = tf.squeeze(tf.where(tf.not_equal(gt, ignore_label)), 1)
    gt = tf.cast(tf.gather(gt, indices), tf.int32)
    prediction = tf.gather(prediction, indices)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)

    loss = tf.reduce_mean(loss)

    return loss

