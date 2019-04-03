"""Training script for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""
import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np

from nets import vgg_16_deeplab_nddr, vgg_16_shortcut_deeplab_nddr
from nets import vgg_16_deeplab_mt, vgg_16_shortcut_deeplab_mt
from nets import vgg_arg_scope

from util.img_encoder_mt import ImageReader
from util.img_decoder import prepare_label
from util.input_arguments import arguments_mt_train

from ops.losses import get_seg_loss, get_normal_loss

from ops.create_train_ops import create_train_ops_mt
from ops.create_summary_ops import create_summary_op_mt

from ops.save_ckpt import save
from ops.load_ckpt import load_mt

slim = tf.contrib.slim

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


def train():
    """Create the model and start the training."""
    args = arguments_mt_train()
    
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    init_weights = map(float, args.init_weights.split(','))
    
    if args.use_random_seed:
        tf.set_random_seed(args.random_seed)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list_1,
            args.data_list_2,
            input_size=input_size,
            random_scale=args.random_scale,
            random_mirror=args.random_mirror,
            random_crop=args.random_crop,
            ignore_label=args.ignore_label,
            img_mean=IMG_MEAN,
            coord=coord,
            task_1=args.task_1,
            task_2=args.task_2)
        image_batch, label_batch_1, label_batch_2 = reader.dequeue(args.batch_size)
    
    # Create network.
    with slim.arg_scope(vgg_arg_scope(weight_decay=args.weight_decay, use_batch_norm=True, is_training=True)):
        if args.network == 'vgg_16_deeplab_nddr':
            _, _, end_points_1, end_points_2, _ = vgg_16_deeplab_nddr(image_batch,
                                                    num_classes_1=args.num_classes_1,
                                                    num_classes_2=args.num_classes_2,
                                                    is_training=True,
                                                    dropout_keep_prob=args.keep_prob,
                                                    init_method=args.init_method,
                                                    init_weights=init_weights)
        elif args.network == 'vgg_16_shortcut_deeplab_nddr':
            _, _, end_points_1, end_points_2, _ = vgg_16_shortcut_deeplab_nddr(image_batch,
                                                    num_classes_1=args.num_classes_1,
                                                    num_classes_2=args.num_classes_2,
                                                    is_training=True,
                                                    dropout_keep_prob=args.keep_prob,
                                                    init_method=args.init_method,
                                                    init_weights=init_weights)
        elif args.network == 'vgg_16_deeplab_mt':
            _, _, end_points_1, end_points_2, _ = vgg_16_deeplab_mt(image_batch,
                                                    num_classes_1=args.num_classes_1,
                                                    num_classes_2=args.num_classes_2,
                                                    is_training=True,
                                                    dropout_keep_prob=args.keep_prob)
        elif args.network == 'vgg_16_shortcut_deeplab_mt':
            _, _, end_points_1, end_points_2, _ = vgg_16_shortcut_deeplab_mt(image_batch,
                                                    num_classes_1=args.num_classes_1,
                                                    num_classes_2=args.num_classes_2,
                                                    is_training=True,
                                                    dropout_keep_prob=args.keep_prob)
        else:
            raise Exception('network name is not recognized!')
    
    # Predictions.
    raw_output_1 = end_points_1['vgg_16_1_6/fc8_voc12']
    raw_output_2 = end_points_2['vgg_16_2_6/fc8_voc12']

    if args.task_1 == 'seg' and args.task_2 == 'normal':
        seg_output = raw_output_1
        seg_classes = args.num_classes_1
        seg_label = prepare_label(label_batch_1, tf.stack(raw_output_1.get_shape()[1:3]), num_classes=seg_classes,
                             one_hot=False, task='seg')
        seg_label_batch = label_batch_1
        seg_loss_scale = args.loss_scale_1

        normal_output = raw_output_2
        normal_classes = args.num_classes_2
        normal_label = prepare_label(label_batch_2, tf.stack(raw_output_2.get_shape()[1:3]), num_classes=normal_classes,
                             one_hot=False, task='normal')
        normal_label_batch = label_batch_2
        normal_loss_scale = args.loss_scale_2

    elif args.task_1 == 'normal' and args.task2 == 'seg':
        seg_output = raw_output_2
        seg_classes = args.num_classes_2
        seg_label = prepare_label(label_batch_2, tf.stack(raw_output_2.get_shape()[1:3]), num_classes=seg_classes,
                                  one_hot=False, task='seg')
        seg_label_batch = label_batch_2
        seg_loss_scale = args.loss_scale_2

        normal_output = raw_output_1
        normal_classes = args.num_classes_1
        normal_label = prepare_label(label_batch_1, tf.stack(raw_output_1.get_shape()[1:3]), num_classes=normal_classes,
                                     one_hot=False, task='normal')
        normal_label_batch = label_batch_1
        normal_loss_scale = args.loss_scale_1

    else:
        raise Exception('check the tasks!')

    seg_loss = get_seg_loss(seg_output, seg_label, seg_classes, args.ignore_label)
    normal_loss = get_normal_loss(normal_output, normal_label, normal_classes, args.ignore_label)
    l2_losses = tf.add_n([args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name])
    reduced_loss = tf.reduce_mean(seg_loss) * seg_loss_scale + tf.reduce_mean(normal_loss) * normal_loss_scale + l2_losses * 0.5
   
    # Train Ops
    train_op, step_ph, learning_rate = create_train_ops_mt(reduced_loss, args)

    # Summaries
    seg_pred = tf.image.resize_bilinear(seg_output, tf.shape(image_batch)[1:3, ])
    seg_pred = tf.argmax(seg_pred, axis=3)
    seg_pred = tf.expand_dims(seg_pred, dim=3)

    normal_pred = tf.image.resize_bilinear(normal_output, tf.shape(image_batch)[1:3, ])

    if 'nddr' in args.network:
        nddr_l2_losses = tf.add_n([args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name and 'nddr' in v.name])
    else:
        nddr_l2_losses = tf.constant(0.0)

    summary_op = create_summary_op_mt(image_batch, seg_label_batch, normal_label_batch, seg_pred, normal_pred, IMG_MEAN,
                                      reduced_loss, seg_loss, normal_loss, l2_losses, nddr_l2_losses,
                                      learning_rate, seg_classes, normal_classes, args.save_num_images, args.network)

    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                           graph=tf.get_default_graph())
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
    
    # Load variables if the checkpoint is provided.
    load_mt(sess, args)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        feed_dict = {step_ph: step}

        if step % args.save_pred_every == 0:
            loss_value, seg_loss_value, normal_loss_value, loss_weight_value, loss_nddr_value, images, labels_1, \
            labels_2, seg_pred_value, normal_pred_value, summary, _ = \
                sess.run([reduced_loss, seg_loss, normal_loss, l2_losses, nddr_l2_losses, image_batch, label_batch_1,
                          label_batch_2, seg_pred, normal_pred, summary_op, train_op], feed_dict=feed_dict)
            save(saver, sess, args.snapshot_dir, step)
        else:
            loss_value, seg_loss_value, normal_loss_value, loss_weight_value, loss_nddr_value, summary, _ = \
                sess.run([reduced_loss, seg_loss, normal_loss, l2_losses, nddr_l2_losses, summary_op, train_op],
                         feed_dict=feed_dict)

        summary_writer.add_summary(summary, step)
        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f} \t seg_loss = {:.3f}, seg_loss_scale = {:.1f} \t normal_loss = {:.3f}, '
              'normal_loss_scale = {:.1f} \t loss_w = {:.3f}, loss_nddr_w = {:.3f} \t nddr_mult = {:.1f}'
              .format(step, loss_value, seg_loss_value, seg_loss_scale, normal_loss_value, normal_loss_scale,
                      loss_weight_value, loss_nddr_value, args.nddr_mult))

    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    train()


