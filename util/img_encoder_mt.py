import os

import numpy as np
import tensorflow as tf


def image_scaling(img, label_1, label_2):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """

    scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label_1 = tf.image.resize_nearest_neighbor(tf.expand_dims(label_1, 0), new_shape)
    label_1 = tf.squeeze(label_1, squeeze_dims=[0])
    label_2 = tf.image.resize_nearest_neighbor(tf.expand_dims(label_2, 0), new_shape)
    label_2 = tf.squeeze(label_2, squeeze_dims=[0])

    return img, label_1, label_2


def image_mirroring(img, label_1, label_2):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """

    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label_1 = tf.reverse(label_1, mirror)
    label_2 = tf.reverse(label_2, mirror)
    return img, label_1, label_2


def random_crop_and_pad_image_and_labels(image, label_1, label_2, crop_h, crop_w, ignore_label=255, task_1='seg',
                                         task_2='normal'):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.concat(axis=2, values=[label_1, label_2])
    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label  # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=2, values=[image, label])
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]),
                                                tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    last_label_1_dim = tf.shape(label_1)[-1]

    combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, tf.shape(combined_pad)[-1]])

    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)

    label_crop_1 = label_crop[:, :, :last_label_1_dim]
    label_crop_2 = label_crop[:, :, last_label_1_dim:]

    # Set static shape so that tensorflow knows shape at compile time. 
    img_crop.set_shape((crop_h, crop_w, 3))
    if task_1 == 'seg':
        label_crop_1.set_shape((crop_h, crop_w, 1))
    elif task_2 == 'normal':
        label_crop_1.set_shape((crop_h, crop_w, 3))
    else:
        raise NotImplementedError

    if task_2 == 'seg':
        label_crop_2.set_shape((crop_h, crop_w, 1))
    elif task_2 == 'normal':
        label_crop_2.set_shape((crop_h, crop_w, 3))
    else:
        raise NotImplementedError

    return img_crop, label_crop_1, label_crop_2


def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line.strip("\n").split(' ')
        except ValueError:  # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
    return images, masks


def read_images_from_disk(input_queue, input_size, random_scale, random_mirror, random_crop, ignore_label, img_mean,
                          task_1='seg', task_2='normal'):  # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.
      
    Returns:
      Two tensors: the decoded image and its mask.
    """

    img_contents = tf.read_file(input_queue[0])
    label_contents_1 = tf.read_file(input_queue[1])
    label_contents_2 = tf.read_file(input_queue[2])

    img = tf.image.decode_image(img_contents, channels=3)
    img.set_shape((None, None, 3))

    # bgr
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)

    # Extract mean.
    img -= img_mean

    if task_1 == 'seg':
        label_1 = tf.image.decode_image(label_contents_1, channels=1)
        label_1.set_shape((None, None, 1))
    elif task_1 == 'normal':
        label_1 = tf.image.decode_image(label_contents_1, channels=3)
        label_1.set_shape((None, None, 3))
    else:
        raise NotImplementedError

    if task_2 == 'seg':
        label_2 = tf.image.decode_image(label_contents_2, channels=1)
        label_2.set_shape((None, None, 1))
    elif task_2 == 'normal':
        label_2 = tf.image.decode_image(label_contents_2, channels=3)
        label_2.set_shape((None, None, 3))
    else:
        raise NotImplementedError

    if input_size is not None:
        h, w = input_size

        # Randomly scale the images and labels.
        if random_scale:
            img, label_1, label_2 = image_scaling(img, label_1, label_2)

        # Randomly mirror the images and labels.
        if random_mirror:
            img, label_1, label_2 = image_mirroring(img, label_1, label_2)

        # Randomly crops the images and labels.
        if random_crop:
            img, label_1, label_2 = random_crop_and_pad_image_and_labels(img, label_1, label_2, h, w, ignore_label,
                                                                         task_1, task_2)
        else:
            img.set_shape((h, w, 3))
            if task_1 == 'seg':
                label_1.set_shape((h, w, 1))
            elif task_1 == 'normal':
                label_1.set_shape((h, w, 3))
            else:
                raise NotImplementedError

            if task_2 == 'seg':
                label_2.set_shape((h, w, 1))
            elif task_2 == 'normal':
                label_2.set_shape((h, w, 3))
            else:
                raise NotImplementedError

    return img, label_1, label_2


class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list_1, data_list_2, input_size,
                 random_scale, random_mirror, random_crop, ignore_label,
                 img_mean, coord, task_1='seg', task_2='normal'):
        '''Initialise an ImageReader.
        
        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          ignore_label: index of label to ignore during the training.
          img_mean: vector of mean colour values.
          coord: TensorFlow queue coordinator.
        '''
        self.data_dir = data_dir
        self.data_list_1 = data_list_1
        self.data_list_2 = data_list_2
        self.input_size = input_size
        self.coord = coord

        image_list_1, self.label_list_1 = read_labeled_image_list(self.data_dir, self.data_list_1)
        image_list_2, self.label_list_2 = read_labeled_image_list(self.data_dir, self.data_list_2)
        assert (image_list_1 == image_list_2)
        self.image_list = image_list_1
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels_1 = tf.convert_to_tensor(self.label_list_1, dtype=tf.string)
        self.labels_2 = tf.convert_to_tensor(self.label_list_2, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels_1, self.labels_2],
                                                   shuffle=input_size is not None)  # not shuffling if it is val
        self.image, self.label_1, self.label_2 = read_images_from_disk(self.queue, self.input_size, random_scale,
                                                                       random_mirror, random_crop, ignore_label,
                                                                       img_mean, task_1, task_2)

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        image_batch, label_batch_1, label_batch_2 = tf.train.batch([self.image, self.label_1, self.label_2],
                                                                   num_elements)
        return image_batch, label_batch_1, label_batch_2
