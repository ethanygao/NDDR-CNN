import tensorflow as tf
from util.img_decoder import inv_preprocess, decode_labels


def create_summary_op_mt(image_batch, seg_label, normal_label, seg_pred, normal_pred, IMG_MEAN,
                         reduced_loss, seg_loss, normal_loss, l2_losses, nddr_l2_losses,
                         learning_rate, seg_classes, normal_classes, save_num_images, network_name):
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # nddr paras summary
    if 'nddr' in network_name:
        nddr_to_train = [v for v in tf.trainable_variables() if 'nddr' in v.name]
        summary_collection_nddr = nddr_to_train
    else:
        summary_collection_nddr = []

    for var in summary_collection_nddr:
        mean = tf.reduce_mean(var)
        summaries.add(tf.summary.scalar(var.op.name + 'mean', mean))
        summaries.add(tf.summary.scalar(var.op.name + 'std', tf.sqrt(tf.reduce_mean(tf.square(var - mean)))))
        summaries.add(tf.summary.scalar(var.op.name + 'max', tf.reduce_max(var)))
        summaries.add(tf.summary.scalar(var.op.name + 'min', tf.reduce_min(var)))
        summaries.add(tf.summary.histogram(var.op.name + 'histogram', var))

    with tf.name_scope('learning'):
        # loss
        summaries.add(tf.summary.scalar('loss', reduced_loss))
        summaries.add(tf.summary.scalar('seg_loss', seg_loss))
        summaries.add(tf.summary.scalar('normal_loss', normal_loss))
        summaries.add(tf.summary.scalar('l2_losses', l2_losses))
        summaries.add(tf.summary.scalar('nddr_l2_losses', nddr_l2_losses))
        # lr
        summaries.add(tf.summary.scalar('learning rate', learning_rate))

    # Image summary.
    summaries.add(tf.summary.image('Input Images',
                                   tf.py_func(inv_preprocess, [image_batch, save_num_images, IMG_MEAN], tf.uint8),
                                   max_outputs=save_num_images))
    with tf.name_scope('Labels'):
        summaries.add(tf.summary.image('Seg Labels',
                                       tf.py_func(decode_labels, 
                                                  [seg_label, save_num_images, seg_classes, 'seg'], tf.uint8),
                                       max_outputs=save_num_images))
        summaries.add(tf.summary.image('Normal Labels',
                                       tf.py_func(decode_labels,
                                                  [normal_label, save_num_images, normal_classes, 'normal'], tf.uint8),
                                       max_outputs=save_num_images))

    with tf.name_scope('Predictions'):
        summaries.add(tf.summary.image('Seg Prediction',
                                       tf.py_func(decode_labels, [seg_pred, save_num_images, seg_classes, 'seg'], tf.uint8),
                                       max_outputs=save_num_images))
        summaries.add(tf.summary.image('Normal Prediction',
                                       tf.py_func(decode_labels, [normal_pred, save_num_images, normal_classes, 'normal'], tf.uint8),
                                       max_outputs=save_num_images))

    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    return summary_op
