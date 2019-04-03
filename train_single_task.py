"""Training script for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""
import time

import tensorflow as tf
import numpy as np

from nets import vgg_arg_scope
from nets import vgg_16_deeplab_st, vgg_16_shortcut_deeplab_st

from util.img_encoder_st import ImageReader
from util.img_decoder import inv_preprocess, decode_labels, prepare_label
from util.input_arguments import arguments_st_train

from ops.losses import get_seg_loss, get_normal_loss

from ops.create_train_ops import create_train_ops_st

from ops.save_ckpt import save
from ops.load_ckpt import load_st


slim = tf.contrib.slim

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


def train():
    """Create the model and start the training."""
    args = arguments_st_train()

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    
    if args.use_random_seed:
        tf.set_random_seed(args.random_seed)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size=input_size,
            random_scale=args.random_scale,
            random_mirror=args.random_mirror,
            random_crop=args.random_crop,
            ignore_label=args.ignore_label,
            img_mean=IMG_MEAN,
            coord=coord,
            task=args.task)
        image_batch, label_batch = reader.dequeue(args.batch_size)
    
    # Create network.
    with slim.arg_scope(vgg_arg_scope(weight_decay=args.weight_decay, use_batch_norm=True, is_training=True)):
        if args.network == 'vgg_16_deeplab_st':
            net, end_points = vgg_16_deeplab_st(image_batch, num_classes=args.num_classes, is_training=True, dropout_keep_prob=args.keep_prob)
        elif args.network == 'vgg_16_shortcut_deeplab_st':
            net, end_points = vgg_16_shortcut_deeplab_st(image_batch, num_classes=args.num_classes, is_training=True, dropout_keep_prob=args.keep_prob)
        else:
            raise Exception('network name is not recognized!')
   
    
    # Predictions.
    raw_output = end_points['vgg_16/fc8_voc12']

    # gt labels
    raw_gt = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=args.num_classes,
                           one_hot=False, task=args.task) # [batch_size, h, w]

    # losses
    if args.task == 'normal':
        loss = get_normal_loss(raw_output, raw_gt, args.num_classes, args.ignore_label) * args.loss_scale
    elif args.task == 'seg':
        loss = get_seg_loss(raw_output, raw_gt, args.num_classes, args.ignore_label) * args.loss_scale

    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)
    
    # Image summary for visualisation.
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, axis=3)
    pred = tf.expand_dims(raw_output_up, dim=3)
    
    images_summary = tf.py_func(inv_preprocess, [image_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [label_batch, args.save_num_images, args.num_classes, args.task], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, args.save_num_images, args.num_classes, args.task], tf.uint8)
    
    total_summary = tf.summary.image('images', 
                                     tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]), 
                                     max_outputs=args.save_num_images) # Concatenate row-wise.
    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                           graph=tf.get_default_graph())
   
    # Define loss and optimisation parameters.
    train_op, step_ph = create_train_ops_st(reduced_loss, args)
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)

    # Load variables if the checkpoint is provided.
    if args.restore_from is not None:
        load_st(sess, args)
    
    # Saver for storing checkpoints of the model.
    save_op = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=args.max_to_keep)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        feed_dict = { step_ph : step }
        
        if step % args.save_pred_every == 0:
            loss_value, images, labels, preds, summary, _ = sess.run([reduced_loss, image_batch, label_batch, pred, total_summary, train_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
            save(save_op, sess, args.snapshot_dir, step)
        else:
            loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    train()
