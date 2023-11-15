"""This File is for Clean Caches"""


import os
import glob
from config.extern_var import settings
from config.settings import Config_Item
from enum import Enum


class clean_caches_type(Enum):
    """定义各种清理类型"""
    total = settings[Config_Item.dataset_cache_path]
    train = settings[Config_Item.train_dataset_cache_path]
    eval = 2
    predict = 3
    train_UNet = 4
    train_type = 5



def clean_caches(clean_type=None):
    """用于清空所有的缓存文件"""

    for file_path in glob.glob(os.path.join(directory_path, '*')):  # 获取目录下所有文件

        if os.path.isfile(file_path):  # 判断文件是否存在

            os.remove(file_path)  # 删除文件

            print(f"{file_path} 文件已被删除")


if __name__ == "__main__":
    clean_caches({clean_caches_type.total})


