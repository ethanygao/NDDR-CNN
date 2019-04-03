import os

import tensorflow as tf
import numpy as np

from PIL import Image

from nets import vgg_16_deeplab_nddr, vgg_16_shortcut_deeplab_nddr
from nets import vgg_16_deeplab_mt, vgg_16_shortcut_deeplab_mt
from nets import vgg_arg_scope

from ops.errors import seg_error, normal_error

from util.img_encoder_mt import ImageReader
from util.img_decoder import inv_preprocess, decode_labels
from util.input_arguments import arguments_mt_eval

slim = tf.contrib.slim

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


def eval():
    """Create the model and start the evaluation process."""
    args = arguments_mt_eval()
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Encode data.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list_1,
            args.data_list_2,
            input_size=None, # No defined input size.
            random_scale=False, # No random scale.
            random_mirror=False, # No random mirror.
            random_crop=False, # No random crop.
            ignore_label=args.ignore_label,
            img_mean=IMG_MEAN,
            coord=coord,
            task_1=args.task_1,
            task_2=args.task_2)
        image, label_1, label_2 = reader.image, reader.label_1, reader.label_2

        image_batch, label_batch_1, label_batch_2 = \
            tf.expand_dims(image, dim=0), tf.expand_dims(label_1, dim=0), tf.expand_dims(label_2, dim=0)  # Add the batch dimension.

    # Create network.
    with slim.arg_scope(vgg_arg_scope(weight_decay=0.0, use_batch_norm=True, is_training=False)):
        if args.network == 'vgg_16_deeplab_nddr':
            _, _, end_points_1, end_points_2, _ = vgg_16_deeplab_nddr(image_batch, 
                                                    num_classes_1=args.num_classes_1,
                                                    num_classes_2=args.num_classes_2,
                                                    is_training=False,
                                                    dropout_keep_prob=1.0)
        elif args.network == 'vgg_16_shortcut_deeplab_nddr':
            _, _, end_points_1, end_points_2, _ = vgg_16_shortcut_deeplab_nddr(image_batch, 
                                                    num_classes_1=args.num_classes_1,
                                                    num_classes_2=args.num_classes_2,
                                                    is_training=False,
                                                    dropout_keep_prob=1.0)
        elif args.network == 'vgg_16_deeplab_mt':
            _, _, end_points_1, end_points_2, _ = vgg_16_deeplab_mt(image_batch,
                                                    num_classes_1=args.num_classes_1,
                                                    num_classes_2=args.num_classes_2,
                                                    is_training=False,
                                                    dropout_keep_prob=1.0)
        elif args.network == 'vgg_16_shortcut_deeplab_mt':
            _, _, end_points_1, end_points_2, _ = vgg_16_shortcut_deeplab_mt(image_batch,
                                                    num_classes_1=args.num_classes_1, 
                                                    num_classes_2=args.num_classes_2, 
                                                    is_training=False, 
                                                    dropout_keep_prob=1.0)
        else:
            raise Exception('network name is not recognized!')
   
    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    raw_output_1 = end_points_1['vgg_16_1_6/fc8_voc12']
    raw_output_2 = end_points_2['vgg_16_2_6/fc8_voc12']

    raw_output_1 = tf.image.resize_bilinear(raw_output_1, tf.shape(image_batch)[1:3,])
    raw_output_2 = tf.image.resize_bilinear(raw_output_2, tf.shape(image_batch)[1:3,])

    if args.task_1 == 'seg' and args.task_2 == 'normal':
        seg_output = raw_output_1
        seg_label = label_batch_1
        seg_classes = args.num_classes_1
        normal_output = raw_output_2
        normal_label = label_batch_2
        normal_classes = args.num_classes_2
    elif args.task_1 == 'normal' and args.task2 == 'seg':
        seg_output = raw_output_2
        seg_label = label_batch_2
        seg_classes = args.num_classes_2
        normal_output = raw_output_1
        normal_label = label_batch_1
        normal_classes = args.num_classes_1
    else:
        raise Exception('check the tasks!')

    save_seg = tf.expand_dims(tf.argmax(seg_output, axis=3), dim=3)
    save_normal = tf.nn.l2_normalize(normal_output, dim=-1) * 255

    if args.save_dir is not None and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # errors:
    mIoU, update_op, correct_pixel, valid_pixel = seg_error(seg_output, seg_label, seg_classes, args.ignore_label)
    cos_distance = normal_error(normal_output, normal_label, normal_classes, args.ignore_label)
    
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

    correct_pixel_all = []
    valid_pixel_all = []
    cos_distance_all = []
    # Iterate over evaluation steps.
    for step in range(args.num_steps):
        imgs, save_segs, save_normals, cd, cp, vp, _ = sess.run([image_batch, save_seg, save_normal, cos_distance, correct_pixel, valid_pixel, update_op])
        correct_pixel_all.append(cp)
        valid_pixel_all.append(vp)
        cos_distance_all.append(cd)
        if step % 100 == 0:
            print('step {:d}'.format(step))

        if args.save_dir is not None:
            org_imgs = inv_preprocess(imgs, 1, IMG_MEAN)
            org_img = org_imgs[0]
            org_img = Image.fromarray(org_img)
            org_img.save(args.save_dir + '/org_' + str(step) + '.png')

            save_segs = decode_labels(save_segs, num_images=1, num_classes=seg_classes, task='seg')
            save_segs = save_segs[0]
            save_segs = Image.fromarray(save_segs)
            save_segs.save(args.save_dir + '/seg_' + str(step) + '.png')

            save_normals = decode_labels(save_normals, num_images=1, num_classes=normal_classes, task='normal')
            save_normals = save_normals[0]
            save_normals = Image.fromarray(save_normals)
            save_normals.save(args.save_dir + '/normal_' + str(step) + '.png')

    mIoU_value = sess.run(mIoU)
    print('Mean IoU: {:.3f}'.format(mIoU_value))
    pixel_acc = sum(correct_pixel_all) / (sum(valid_pixel_all) + 0.0)
    print('Pixel Acc: {:.3f}'.format(pixel_acc))

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
        f.write(checkpoint_path.split('/')[-1] + '     Mean IoU: {:.3f} \n'.format(mIoU_value))
        f.write(checkpoint_path.split('/')[-1] + '     Pixel Acc: {:.3f} \n'.format(pixel_acc))
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
