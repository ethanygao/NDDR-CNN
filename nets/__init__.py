# VGG-Deeplab models:
# arg scope
from model_arg_scopes import vgg_arg_scope
# single task baselines
from vgg_deeplab_st_baseline import vgg_16_deeplab_st, vgg_16_shortcut_deeplab_st
# multi task baselines
from vgg_deeplab_mt_baseline import vgg_16_deeplab_mt, vgg_16_shortcut_deeplab_mt
# NDDR models
from vgg_deeplab_nddr import vgg_16_deeplab_nddr, vgg_16_shortcut_deeplab_nddr