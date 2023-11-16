"""This File is for Clean Caches"""


import os
import glob
from config.extern_var import settings
from config.settings import Config_Item
from enum import Enum
from data_coding.data_cache import dataset_cache_type


def clean_caches(clean_types=None):
    """用于清空指定的的缓存文件"""
    print(f"\nNow Clean All Caches under {clean_types}")
    for clean_type in clean_types:
        for cache_path in glob.glob(os.path.join(clean_type, '**/*'), recursive=True):  # 获取目录下所有文件
            if os.path.isdir(cache_path):
                print(f"remove all files under {cache_path}")
            elif os.path.isfile(cache_path):  # 判断文件是否存在
                os.remove(cache_path)  # 删除文件


if __name__ == "__main__":
    clean_caches((dataset_cache_type.total.value,
                  ))


