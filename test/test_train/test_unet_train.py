"""用于训练UNet网络"""

from train.train_unet import *
from data_coding.data_cache import *


def test_train_unet():
    """Unet训练的基本测试"""
    train_unet = Train_UNet()
    train_unet.train_all_epochs()
