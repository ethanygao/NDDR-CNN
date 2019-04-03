from PIL import Image
import numpy as np
import tensorflow as tf

label_colours = [(178, 45, 45), (153, 115, 115), (64, 36, 32), (255, 68, 0), (89, 24, 0), (191, 121, 96), (191, 102, 0),
                 (76, 41, 0), (153, 115, 38), (102, 94, 77), (242, 194, 0), (191, 188, 143), (226, 242, 0),
                 (119, 128, 0), (59, 64, 0), (105, 191, 48), (81, 128, 64), (0, 255, 0), (0, 51, 7), (191, 255, 208),
                 (96, 128, 113), (0, 204, 136), (13, 51, 43), (0, 191, 179), (0, 204, 255), (29, 98, 115), (0, 34, 51),
                 (163, 199, 217), (0, 136, 255), (41, 108, 166), (32, 57, 128), (0, 22, 166), (77, 80, 102),
                 (119, 54, 217), (41, 0, 77), (222, 182, 242), (103, 57, 115), (247, 128, 255), (191, 0, 153),
                 (128, 96, 117), (127, 0, 68), (229, 0, 92), (76, 0, 31), (255, 128, 179), (242, 182, 198)]


def decode_labels(mask, num_images=1, num_classes=21, task='seg'):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w, c = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        if task == 'normal':
            outputs[i] = mask[i]
        elif task == 'seg':
            img = Image.new('RGB', (w, h), (255, 255, 255))  # unlabeled part is white (255, 255, 255)
            pixels = img.load()
            for j_, j in enumerate(mask[i, :, :, 0]):
                for k_, k in enumerate(j):
                    if k < num_classes:
                        pixels[k_, j_] = label_colours[k]
            outputs[i] = np.array(img)
        else:
            raise Exception('task name is not recognized!')

    return outputs


def inv_preprocess(imgs, num_images, img_mean):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    n, h, w, c = imgs.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (imgs[i] + img_mean)[:, :, ::-1].astype(np.uint8)
    return outputs


def prepare_label(input_batch, new_size, num_classes, one_hot=True, task='seg'):
    """Resize masks and perform one-hot encoding.
    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.
      num_classes: number of classes to predict (including background).
      one_hot: whether perform one-hot encoding.
    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        if task == 'seg':
            input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
    return input_batch