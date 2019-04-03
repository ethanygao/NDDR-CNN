import os

import tensorflow as tf
import numpy as np

from PIL import Image

from nets import vgg_arg_scope
from nets import vgg_16_deeplab_st, vgg_16_shortcut_deeplab_st

from ops.errors import seg_error, normal_error

from util.img_encoder_st import ImageReader
from util.img_decoder import inv_preprocess, decode_labels
from util.input_arguments import arguments_st_eval

slim = tf.contrib.slim

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def eval():
    """Create the model and start the evaluation process."""
    args = arguments_st_eval()
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Encode the data.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size=None, # No defined input size.
            random_scale=False, # No random scale.
            random_mirror=False, # No random mirror.
            random_crop=False, # No random crop.
            ignore_label=args.ignore_label,
            img_mean=IMG_MEAN,
            coord=coord,
            task=args.task)
        image, label = reader.image, reader.label

    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.

    # Create network.
    with slim.arg_scope(vgg_arg_scope(weight_decay=0.0, use_batch_norm=True, is_training=False)):
        if args.network == 'vgg_16_deeplab_st':
            net, end_points = vgg_16_deeplab_st(image_batch, num_classes=args.num_classes, is_training=False, dropout_keep_prob=1.0)
        elif args.network == 'vgg_16_shortcut_deeplab_st':
            net, end_points = vgg_16_shortcut_deeplab_st(image_batch, num_classes=args.num_classes, is_training=False, dropout_keep_prob=1.0)
        else:
            raise Exception('network name is not recognized!')
   
    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    raw_output = end_points['vgg_16/fc8_voc12']
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])

    if args.task == 'normal':
        save_pred = tf.nn.l2_normalize(raw_output, dim=-1) * 255
        cos_distance = normal_error(raw_output, label_batch, args.num_classes, args.ignore_label)
    elif args.task == 'seg':
        save_pred = tf.expand_dims(tf.argmax(raw_output, axis=3), dim=3)
        mIoU, update_op, correct_pixel, valid_pixel = seg_error(raw_output, label_batch, args.num_classes, args.ignore_label)
    else:
        raise Exception('task name is not recognized!')    
    
    # Save folder
    if args.save_dir is not None and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    if args.restore_from is not None:
        if tf.gfile.IsDirectory(args.restore_from):
            folder_name = args.restore_from
            checkpoint_path = tf.train.latest_checkpoint(args.restore_from)
        else:
            folder_name = args.restore_from.replace(args.restore_from.split('/')[-1], '')
            checkpoint_path = args.restore_from

        tf.train.Saver(var_list=restore_var).restore(sess, checkpoint_path)
        print("Restored model parameters from {}".format(checkpoint_path))
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    # Iterate over evaluation steps.
    correct_pixel_all = []
    valid_pixel_all = []
    cos_distance_all = []
    for step in range(args.num_steps):
        if args.task == 'seg':
            imgs, save_preds, cp, vp, _ = sess.run([image_batch, save_pred, correct_pixel, valid_pixel, update_op])
            correct_pixel_all.append(cp)
            valid_pixel_all.append(vp)
        elif args.task == 'normal':
            imgs, save_preds, cd = sess.run([image_batch, save_pred, cos_distance])
            cos_distance_all.append(cd)
        else:
            raise Exception('task name is not recognized!')

        if step % 100 == 0:
            print('step {:d}'.format(step))

        if args.save_dir is not None:
            org_imgs = inv_preprocess(imgs, 1, IMG_MEAN)
            org_img = org_imgs[0]
            org_img = Image.fromarray(org_img)
            org_img.save(args.save_dir + '/org_' + str(step) + '.png')

            save_imgs = decode_labels(save_preds, num_images=1, num_classes=args.num_classes, task=args.task)
            save_img = save_imgs[0]
            save_img = Image.fromarray(save_img)
            save_img.save(args.save_dir + '/save_' + str(step) + '.png')

    if args.task == 'seg':
        mIoU_value = sess.run(mIoU)
        print('Mean IoU: {:.3f}'.format(mIoU_value))
        pixel_acc = sum(correct_pixel_all) / (sum(valid_pixel_all) + 0.0)
        print('Pixel Acc: {:.3f}'.format(pixel_acc))
        with open(folder_name + '/results.txt', 'a') as f:
            f.write(checkpoint_path.split('/')[-1] + '     Mean IoU: {:.3f} \n'.format(mIoU_value))
            f.write(checkpoint_path.split('/')[-1] + '     Pixel Acc: {:.3f} \n'.format(pixel_acc))

    elif args.task == 'normal':
        cos_distance_all = np.concatenate(cos_distance_all, axis=0)
        cosine_distance = np.minimum(np.maximum(cos_distance_all, -1.0), 1.0)
        angles = np.arccos(cosine_distance) / np.pi * 180.0
        print('Mean: {:.3f}'.format(np.mean(angles)))
        print('Median: {:.3f}'.format(np.median(angles)))
        print('RMSE: {:.3f}'.format(np.sqrt(np.mean(angles ** 2))))
        print('11.25: {:.3f}'.format(np.mean(np.less_equal(angles, 11.25)) * 100))
        print('22.5: {:.3f}'.format(np.mean(np.less_equal(angles, 22.5)) * 100))
        print('30: {:.3f}'.format(np.mean(np.less_equal(angles, 30.0)) * 100))
        print('45: {:.3f}'.format(np.mean(np.less_equal(angles, 45.0)) * 100))
        with open(folder_name + '/results.txt', 'a') as f:
            f.write(checkpoint_path.split('/')[-1] + '     Mean: {:.3f} \n'.format(np.mean(angles)))
            f.write(checkpoint_path.split('/')[-1] + '     Median: {:.3f} \n'.format(np.median(angles)))
            f.write(checkpoint_path.split('/')[-1] + '     RMSE: {:.3f} \n'.format(np.sqrt(np.mean(angles ** 2))))
            f.write(checkpoint_path.split('/')[-1] + '     11.25: {:.3f} \n'.format(np.mean(np.less_equal(angles, 11.25)) * 100))
            f.write(checkpoint_path.split('/')[-1] + '     22.5: {:.3f} \n'.format(np.mean(np.less_equal(angles, 22.5)) * 100))
            f.write(checkpoint_path.split('/')[-1] + '     30: {:.3f} \n'.format(np.mean(np.less_equal(angles, 30.0)) * 100))
            f.write(checkpoint_path.split('/')[-1] + '     45: {:.3f} \n'.format(np.mean(np.less_equal(angles, 40.0)) * 100))

    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    eval()
