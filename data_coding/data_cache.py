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
import utils.abort as abort


class dataset_cache_type(Enum):
    """以路径方式简明定义各种cache类型"""
    total = settings[Config_Item.dataset_cache]

    train = settings[Config_Item.train_dataset_cache]
    eval = settings[Config_Item.eval_dataset_cache]
    predict = settings[Config_Item.predict_dataset_cache]

    train_UNet = settings[Config_Item.train_UNet_dataset_cache]
    train_UNet_input = settings[Config_Item.train_UNet_input_dataset_cache]
    train_UNet_label = settings[Config_Item.train_UNet_label_dataset_cache]
    train_type = settings[Config_Item.train_type_dataset_cache]
    train_type_annotated = settings[Config_Item.train_type_annotated_dataset_cache]
    train_type_unannotated = settings[Config_Item.train_type_unannotated_dataset_cache]

    eval_UNet = settings[Config_Item.eval_UNet_dataset_cache]
    eval_UNet_input = settings[Config_Item.eval_UNet_input_dataset_cache]
    eval_UNet_label = settings[Config_Item.eval_UNet_label_dataset_cache]
    eval_type = settings[Config_Item.eval_type_dataset_cache]
    eval_type_annotated = settings[Config_Item.eval_type_annotated_dataset_cache]
    eval_type_unannotated = settings[Config_Item.eval_type_unannotated_dataset_cache]

    predict_raw = settings[Config_Item.predict_raw_dataset_cache]
    predict_raw_annotated = settings[Config_Item.predict_raw_annotated_dataset_cache]
    predict_raw_unannotated = settings[Config_Item.predict_raw_unannotated_dataset_cache]
    predict_UNet = settings[Config_Item.predict_UNet_dataset_cache]
    predict_UNet_annotated = settings[Config_Item.predict_UNet_annotated_dataset_cache]
    predict_UNet_unannotated = settings[Config_Item.predict_UNet_unannotated_dataset_cache]


def from_cache_type_to_parameter(cache_type):
    """将cache_type转化为路径"""
    if type(cache_type) == dataset_cache_type:
        return cache_type.value[0], cache_type.value[1]
    elif type(cache_type) == dict:
        return cache_type[0], cache_type[1]
    elif type(cache_type) == str:
        return cache_type, 0
    else:
        raise abort.CacheAbort("Error Cache Type!!!")


@functools.lru_cache(maxsize=settings[Config_Item.cache_maxsize], typed=True)
def Get_CT_Cache(index, cache_type):
    """用缓存方式获取最终可直接使用的张量"""
    path = from_cache_type_to_parameter(cache_type)[0]
    index_str = "{:06}".format(index)
    cache_file = os.path.join(path, index_str)
    return joblib.load(cache_file)


def Flush_CT_Data_To_Mem(cache_types):
    """将所有缓存刷入内存中，cache_types本质上是Config_Item"""
    print(f"\nFlush {cache_types} CT Data To Mem:")
    for cache_type in cache_types:
        path, num = from_cache_type_to_parameter(cache_type)
        print(f"\nFlush {cache_type} CT Data To Mem:")
        counter = DynamicCounter(num, f"{path}", 100)
        for i in range(num):
            Get_CT_Cache(i, path)
    print()


def Save_CT_Candidate(index, cache_type, data, *arg, **kwargs):
    """存储可用CT数据的缓存"""
    path = from_cache_type_to_parameter(cache_type)[0]
    data = data.to(torch.device("cpu"))
    index_str = "{:06}".format(index)
    cache_file = os.path.join(path, index_str)
    joblib.dump(data, cache_file)
