import tensorflow as tf

def normal_error(prediction, gt, num_classes, ignore_label):
    
    # normalized, ignored gt and prediction
    prediction = tf.reshape(prediction, [-1, num_classes])
    gt = tf.reshape(gt, [-1, num_classes])

    mask = tf.reduce_any(tf.not_equal(gt, ignore_label), axis=-1)
    indices = tf.squeeze(tf.where(tf.equal(mask, True)), axis=-1)

    gt = tf.cast(tf.gather(gt, indices), tf.int32)
    prediction = tf.gather(prediction, indices)

    gt = tf.nn.l2_normalize(tf.cast(gt, tf.float32), dim=-1)
    prediction = tf.nn.l2_normalize(prediction, dim=-1)

    # cosine distance
    cos_distance = tf.reduce_sum(prediction * gt, axis=-1)

    return cos_distance


def seg_error(prediction, gt, num_classes, ignore_label):
    prediction = tf.argmax(prediction, axis=3)
    prediction = tf.expand_dims(prediction, dim=3)

    prediction = tf.reshape(prediction, [-1,])
    gt = tf.reshape(gt, [-1,])

    valid_classes = tf.less(gt, num_classes)
    valid_pixels = tf.not_equal(gt, ignore_label)
    valid_all = tf.logical_and(valid_classes, valid_pixels)
    indices = tf.squeeze(tf.where(tf.equal(valid_all, True)), axis=-1)

    gt = tf.cast(tf.gather(gt, indices), tf.int32)
    prediction = tf.cast(tf.gather(prediction, indices), tf.int32)

    # mIoU
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(prediction, gt, num_classes=num_classes)

    # correct pixels
    correct_pixel = tf.reduce_sum(tf.cast(tf.equal(prediction, gt), tf.int32))
    valid_pixel = tf.size(prediction)

    return mIoU, update_op, correct_pixel, valid_pixel