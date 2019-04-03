import argparse

def arguments_st_eval():
    # Arguments for single task evaluation

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data-dir", type=str, default='datasets/nyu_v2',
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default='datasets/nyu_v2/list/testing_seg.txt',
                        help="Path to the file listing the images in the dataset. {testing_seg.txt, testing_normal_mask.txt}")
    parser.add_argument("--restore-from", type=str, default=None,
                        help="Where restore model parameters from.")

    # Model
    parser.add_argument("--network", type=str, default='vgg_16_deeplab_st',
                        help="Which network is used for training {vgg_16_deeplab_st, vgg_16_shortcut_deeplab_st}")
    parser.add_argument("--num-classes", type=int, default=40,
                        help="Number of classes to predict. {40 for segmentation, 3 for surface normal estimation}")
    parser.add_argument("--num-steps", type=int, default=654,
                        help="Number of images in the validation set.")

    # Task
    parser.add_argument("--task", type=str, default='seg',
                        help="Which task is evaluated {seg, normal}")

    # Preprocessing
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="The index of the label to ignore during the training.")

    # Saving
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Where to save the predicted images")

    return parser.parse_args()


def arguments_mt_eval():
    # Arguments for multi-task evaluation
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data-dir", type=str, default='datasets/nyu_v2',
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list-1", type=str, default='datasets/nyu_v2/list/testing_seg.txt',
                        help="Path to the file listing the images in the dataset for task 1.")
    parser.add_argument("--data-list-2", type=str, default='datasets/nyu_v2/list/testing_normal_mask.txt',
                        help="Path to the file listing the images in the dataset for task 2.")
    parser.add_argument("--restore-from", type=str, default=None,
                        help="Where restore model parameters from.")

    # Model
    parser.add_argument("--network", type=str, default='vgg_16_deeplab_nddr',
                        help="Which network is used for training {vgg_16_deeplab_nddr, vgg_16_shortcut_deeplab_nddr, vgg_16_deeplab_mt, vgg_16_shortcut_deeplab_mt}")
    parser.add_argument("--num-classes-1", type=int, default=40,
                        help="Number of classes to predict, for task 1.")
    parser.add_argument("--num-classes-2", type=int, default=3,
                        help="Number of classes to predict, for task 2.")
    parser.add_argument("--num-steps", type=int, default=654,
                        help="Number of images in the validation set.")

    # Task
    parser.add_argument("--task-1", type=str, default='seg',
                        help="Which task is evaluated {seg, normal}")
    parser.add_argument("--task-2", type=str, default='normal',
                        help="Which task is evaluated {seg, normal}")

    # Preprocessing
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="The index of the label to ignore during the training.")

    # Saving
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Where to save the predicted images")

    return parser.parse_args()


def arguments_st_train():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data-dir", type=str, default='datasets/nyu_v2',
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default='datasets/nyu_v2/list/training_seg.txt',
                        help="Path to the file listing the images in the dataset. {training_seg.txt, training_normal_mask.txt}")
    parser.add_argument("--restore-from", type=str, default='pretrained/vgg_lfov/model.ckpt-init-slim',
                        help="Where restore model parameters from.")
    parser.add_argument("--checkpoint-exclude", type=str, default='fc8',
                        help="Variables that contain --checkpoint-exclude is not restored.")

    # Model
    parser.add_argument("--network", type=str, default='vgg_16_deeplab_st',
                        help="Which network is used for training {vgg_16_deeplab_st, vgg_16_shortcut_deeplab_st}")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-classes", type=int, default=40,
                        help="Number of classes to predict. {40 for segmentation, 3 for surface normal estimation}")

    # Task
    parser.add_argument("--task", type=str, default='seg',
                        help="Which task is evaluated {seg, normal}")

    # Preprocessing
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default='321,321',
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--random-mirror", type=bool, default=True,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", type=bool, default=True,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-crop", type=bool, default=True,
                        help="Whether to randomly scale the inputs during the training.")

    # Training
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--keep-prob", type=float, default=0.5,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--loss-scale", type=float, default=1.0,
                        help="The scale that multiples loss")
    parser.add_argument("--num-steps", type=int, default=20001,
                        help="Number of training steps.")

    # Random Seed
    parser.add_argument("--use-random-seed", type=bool, default=True,
                        help="Whether to use random seed.")
    parser.add_argument("--random-seed", type=int, default=1234,
                        help="Random seed to have reproducible results.")

    # Saving
    parser.add_argument("--save-num-images", type=int, default=2,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=5000,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--max-to-keep", type=int, default=2,
                        help="How many checkpoints to keep.")
    parser.add_argument("--snapshot-dir", type=str, default='save/seg/',
                        help="Where to save snapshots of the model.")

    return parser.parse_args()


def arguments_mt_train():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data-dir", type=str, default='datasets/nyu_v2',
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list-1", type=str, default='datasets/nyu_v2/list/training_seg.txt',
                        help="Path to the file listing the images in the dataset for task 1.")
    parser.add_argument("--data-list-2", type=str, default='datasets/nyu_v2/list/training_normal_mask.txt',
                        help="Path to the file listing the images in the dataset for task 2.")
    parser.add_argument("--restore-from-1", type=str, default='pretrained/vgg_lfov/model.ckpt-init-slim',
                        help="Where restore model parameters from for task 1.")
    parser.add_argument("--restore-from-2", type=str, default='pretrained/vgg_lfov/model.ckpt-init-slim',
                        help="Where restore model parameters from for task 2.")
    parser.add_argument("--checkpoint-exclude", type=str, default='fc8',
                        help="Variables that contain --checkpoint-exclude is not restored.")
    parser.add_argument("--replace-from", type=str, default='vgg_16',
                        help="The name to replace from. (because the weights to load has different variable names)")
    parser.add_argument("--replace-to-list", type=str, default='vgg_16_1,vgg_16_2',
                        help="The name to replace to, for tasks 1 and 2. (because the weights to load has different variable names)")

    # Model
    parser.add_argument("--network", type=str, default='vgg_16_deeplab_nddr',
                        help="Which network is used for training {vgg_16_deeplab_nddr, vgg_16_shortcut_deeplab_nddr, vgg_16_deeplab_mt, vgg_16_shortcut_deeplab_mt}")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-classes-1", type=int, default=40,
                        help="Number of classes to predict, for task 1.")
    parser.add_argument("--num-classes-2", type=int, default=3,
                        help="Number of classes to predict, for task 2.")
    parser.add_argument("--init-method", type=str, default='constant',
                        help="initializing method for weights of NDDR layers. {'constant, xavier'}")
    parser.add_argument("--init-weights", type=str, default='0.9,0.1',
                        help="initializing weights for NDDR layers if init_method is 'constant'")

    # Task
    parser.add_argument("--task-1", type=str, default='seg',
                        help="Which task is evaluated {seg, normal}")
    parser.add_argument("--task-2", type=str, default='normal',
                        help="Which task is evaluated {seg, normal}")

    # Preprocessing
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default='321,321',
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--random-mirror", action="store_false",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_false",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-crop", action="store_false",
                        help="Whether to randomly scale the inputs during the training.")

    # Training
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--keep-prob", type=float, default=0.5,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--loss-scale-1", type=float, default=1.0,
                        help="The scale that multiples loss 1")
    parser.add_argument("--loss-scale-2", type=float, default=20.0,
                        help="The scale that multiples loss 2")
    parser.add_argument("--nddr-mult", type=float, default=100,
                        help="The scale that multiples learning rate on nddr paras")
    parser.add_argument("--num-steps", type=int, default=20001,
                        help="Number of training steps.")

    # Random Seed
    parser.add_argument("--use-random-seed", action="store_false",
                        help="Whether to use random seed.")
    parser.add_argument("--random-seed", type=int, default=1234,
                        help="Random seed to have reproducible results.")

    # Saving
    parser.add_argument("--save-num-images", type=int, default=2,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=5000,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--max-to-keep", type=int, default=2,
                        help="How many checkpoints to keep.")
    parser.add_argument("--snapshot-dir", type=str, default='save/seg_normal/',
                        help="Where to save snapshots of the model.")

    return parser.parse_args()
