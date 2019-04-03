# NDDR-CNN, A Simple and General Architecture for General-Purpose CNN based Multi-Task Learning

![NDDR-CNN](https://github.com/ethanygao/NDDR-CNN/blob/master/README/NDDR-CNN.png)

<!-- ![NDDR-CNN-Shortcut](README/NDDR-CNN-Shortcut.png =200x) -->

____

Codes for ``NDDR-CNN: Layerwise Feature Fusing in Multi-Task CNNs by Neural Discriminative Dimensionality Reduction''.

NDDR-CNN is a **simple and general architecture for general-purpose CNN based multi-task learning**, which automatically learns feature fusing (i.e., sharing and splitting) at every CNN layer from different tasks. This is enabled by Neural Discriminative Dimensionality Reduction (NDDR), consisting of simple **1x1 convolution, Batch Normalization, and Weight Decay**.

<!-- The proposed NDDR-CNN architecture can be **extended to various state-of-the-art CNN architectures in a “plug-and-play” manner with end-to-end training**. -->

This repository illustrates example experiments (including ablations) with **VGG-16** and **VGG-16-Shortcut** for **semantic segmentation** and **surface normal prediction** on NYUv2 dataset. 

Extension to **other state-of-the-art CNN architectures** and/or **other tasks** is similar and can be done straightforwardly in a **“plug-and-play”** manner.

Please refer to our paper for more technical details:
>Yuan Gao, Jiayi Ma, Mingbo Zhao, Wei Liu, and Alan L. Yuille. NDDR-CNN: Layerwise Feature Fusing in Multi-Task CNNs by Neural Discriminative Dimensionality Reduction, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019. [[Arxiv]](https://arxiv.org/abs/1801.08297)

## Usage

### Prerequisites
+ Python 2
+ tensorflow (tested on 1.4.0 and 1.4.1), PIL, numpy, argparse. These can be installed by:
```sh
$ pip install pillow, numpy, argparse, tensorflow==1.4.0
```

### Dataset
First, prepare dataset and pretrained weights for the single task networks. Deeplab-VGG16 model and NYU v2 dataset are used as examples in this repository. You can download the [official Caffe Deeplab-VGG16 model](http://liangchiehchen.com/projects/DeepLab.html) then converting it into tensorflow format with [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow), and [the official NYU v2 dataset](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) and [the surface normal labels](https://cs.nyu.edu/~deigen/dnl/normals_gt.tgz) then formating them into specific separated folders with a list file indicating training/testing.

**To make life easier, you can alternatively download the formatted datasets and the pretrained weights [here](https://www.dropbox.com/sh/e44jyh6ayuimigp/AADHlrCVnCDyTdDT9wDOy8cUa?dl=0), then unzip them in the current folder.**

When you are all set, you should have the following sub-folds:
```
datasets/nyu_v2/list
datasets/nyu_v2/nyu_train_val
models/vgg_deeplab_lfov
models/nyu_v2/slim_finetune_seg
models/nyu_v2/slim_finetune_normal
```

### Training
All the arguments to train/eval an NDDR-CNN are shown in `util/input_arguments.py`.

All the training scripts reproducing the results in Tables 1-4, 6 of our paper are in `experiments` folder, including ablation analysis and experiments on VGG-16, experiments on VGG-16-shortcut.

For example, you can simply use the following command to train an NDDR-CNN with VGG-16-shortcut architecture:
```sh
$ CUDA_VISIBLE_DEVICES=0 sh experiments/VGG16-shortcut/NDDR.sh
```

### Evaluation
All the arguments to train/eval an NDDR-CNN are shown in `util/input_arguments.py`.

For example, to evaluate the model trained on VGG-16-shortcut architecture in the previous section, you can type:
```sh
$ CUDA_VISIBLE_DEVICES=0 python eval_multiple_tasks.py \
    --restore-from=save/VGG16_shortcut/NDDR \
    --task-1=seg --task-2=normal --num-classes-1=40 --num-classes-2=3 \
    --network=vgg_16_shortcut_deeplab_nddr
```

## Contacts
For questions about the code or the paper, feel free to contact me by ethan.y.gao@gmail.com.

## Bibtex
If this code is helpful to your research, please consider citing [our paper](https://arxiv.org/abs/1801.08297) by:
```
@inproceedings{gao2019nddr,
  title={NDDR-CNN: Layerwise Feature Fusing in Multi-Task CNNs by Neural Discriminative Dimensionality Reduction},
  author={Gao, Yuan and Ma, Jiayi and Zhao, Mingbo and Liu, Wei and Yuille, Alan L},
  booktitle = {CVPR},
  year={2019}
}
```

## Acknowledgments
This project refers to the code from [tf-slim models](https://github.com/tensorflow/models/tree/master/research/slim) and [tensorflow-deeplab-resnet](https://github.com/DrSleep/tensorflow-deeplab-resnet).