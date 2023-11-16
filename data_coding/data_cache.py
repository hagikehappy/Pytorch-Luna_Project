"""数据存取类具体函数"""


import os
import joblib
import functools
import config.extern_var as extern_var
from config.extern_var import settings
from config.settings import Config_Item
from utils.timer import *
import torch
from enum import Enum


class dataset_cache_type(Enum):
    """以路径方式简明定义各种cache类型"""
    total = settings[Config_Item.dataset_cache_path]

    train = settings[Config_Item.train_dataset_cache_path]
    eval = settings[Config_Item.eval_dataset_cache_path]
    predict = settings[Config_Item.predict_dataset_cache_path]

    train_UNet = settings[Config_Item.train_UNet_dataset_num]
    train_UNet_input = settings[Config_Item.train_UNet_input_dataset_cache_path]
    train_UNet_label = settings[Config_Item.train_UNet_label_dataset_cache_path]
    train_type = settings[Config_Item.train_type_dataset_cache_path]
    train_type_annotated = settings[Config_Item.train_type_annotated_dataset_cache_path]
    train_type_unannotated = settings[Config_Item.train_type_unannotated_dataset_cache_path]

    eval_UNet = settings[Config_Item.eval_UNet_dataset_num]
    eval_UNet_input = settings[Config_Item.eval_UNet_input_dataset_cache_path]
    eval_UNet_label = settings[Config_Item.eval_UNet_label_dataset_cache_path]
    eval_type = settings[Config_Item.eval_type_dataset_cache_path]
    eval_type_annotated = settings[Config_Item.eval_type_annotated_dataset_cache_path]
    eval_type_unannotated = settings[Config_Item.eval_type_unannotated_dataset_cache_path]

    predict_raw = settings[Config_Item.predict_raw_dataset_cache_path]
    predict_raw_annotated = settings[Config_Item.predict_raw_annotated_dataset_cache_path]
    predict_raw_unannotated = settings[Config_Item.predict_raw_unannotated_dataset_cache_path]
    predict_UNet = settings[Config_Item.predict_UNet_dataset_cache_path]
    predict_UNet_annotated = settings[Config_Item.predict_UNet_annotated_dataset_cache_path]
    predict_UNet_unannotated = settings[Config_Item.predict_UNet_unannotated_dataset_cache_path]


@functools.lru_cache(maxsize=settings[Config_Item.cache_maxsize], typed=True)
def Get_CT_Cache(index, path):
    """用缓存方式获取最终可直接使用的张量"""
    index_str = "{:06}".format(index)
    cache_file = os.path.join(path, index_str)
    return joblib.load(cache_file)


def Flush_All_CT_Data_To_Mem():
    """将所有缓存刷入内存中"""
    print("\nFlush_All_CT_Data_To_Mem:")
    counter = DynamicCounter(extern_var.CACHE_MAXSIZE_ANNOTED, "CT_Annoted_INPUT", 100)

    print()


def Save_CT_Candidate(index, path, data, *arg, **kwargs):
    """存储可用CT数据的缓存"""
    data = data.to(torch.device("cpu"))
    index_str = "{:06}".format(index)
    cache_file = os.path.join(path, index_str)
    joblib.dump(data, cache_file)
