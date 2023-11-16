"""数据存取类具体函数"""


import os
import joblib
import functools
import config.extern_var as extern_var
from tools.tool import *
import torch


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


def Get_CT_Candidate(index, path):
    """用缓存封装的获取最终可用于处理图片的方式"""
    index_str = "{:06}".format(index)
    cache_file = os.path.join(path, index_str)
    return joblib.load(cache_file)

@functools.lru_cache(maxsize=extern_var.CACHE_MAXSIZE_ANNOTED, typed=True)
def Get_CT_Annoted_INPUT(index):
    """从内存缓存获取"""
    index_str = "{:06}".format(index)
    cache_file = os.path.join(extern_var.CACHE_PATH_ANNOTED_INPUT, index_str)
    return joblib.load(cache_file)

@functools.lru_cache(maxsize=extern_var.CACHE_MAXSIZE_ANNOTED, typed=True)
def Get_CT_Annoted_LABEL(index):
    """从内存缓存获取"""
    index_str = "{:06}".format(index)
    cache_file = os.path.join(extern_var.CACHE_PATH_ANNOTED_LABEL, index_str)
    return joblib.load(cache_file)

@functools.lru_cache(maxsize=extern_var.CACHE_MAXSIZE_ANNOTED, typed=True)
def Get_CT_Annoted_RAW(index):
    """从内存缓存获取"""
    index_str = "{:06}".format(index)
    cache_file = os.path.join(extern_var.CACHE_PATH_ANNOTED_RAW, index_str)
    return joblib.load(cache_file)

@functools.lru_cache(maxsize=extern_var.CACHE_MAXSIZE_UNANNOTED, typed=True)
def Get_CT_Unannotated_OUTPUT(index):
    """从内存缓存获取"""
    index_str = "{:06}".format(index)
    cache_file = os.path.join(extern_var.CACHE_PATH_UNANNOTED_OUTPUT, index_str)
    return joblib.load(cache_file)

@functools.lru_cache(maxsize=extern_var.CACHE_MAXSIZE_UNANNOTED, typed=True)
def Get_CT_Unannotated_RAW(index):
    """从内存缓存获取"""
    index_str = "{:06}".format(index)
    cache_file = os.path.join(extern_var.CACHE_PATH_UNANNOTED_RAW, index_str)
    return joblib.load(cache_file)

def Flush_All_CT_Data_To_Mem():
    """将所有缓存刷入内存中"""
    print("\nFlush_All_CT_Data_To_Mem:")
    counter = DynamicCounter(extern_var.CACHE_MAXSIZE_ANNOTED, "CT_Annoted_INPUT", 100)
    for i in range(extern_var.CACHE_MAXSIZE_ANNOTED):
        Get_CT_Annoted_INPUT(i)
        counter.increment()
    counter = DynamicCounter(extern_var.CACHE_MAXSIZE_ANNOTED, "CT_Annoted_LABEL", 100)
    for i in range(extern_var.CACHE_MAXSIZE_ANNOTED):
        Get_CT_Annoted_LABEL(i)
        counter.increment()
    counter = DynamicCounter(extern_var.CACHE_MAXSIZE_ANNOTED, "CT_Annoted_RAW", 100)
    for i in range(extern_var.CACHE_MAXSIZE_ANNOTED):
        Get_CT_Annoted_RAW(i)
        counter.increment()
    # counter = DynamicCounter(EXTERN_VAR.CACHE_MAXSIZE_UNANNOTED, "CT_Unannotated_OUTPUT", 100)
    # for i in range(EXTERN_VAR.CACHE_MAXSIZE_UNANNOTED):
    #     Get_CT_Unannotated_OUTPUT(i)
    #     counter.increment()
    # counter = DynamicCounter(EXTERN_VAR.CACHE_MAXSIZE_UNANNOTED, "CT_Unannotated_RAW", 100)
    # for i in range(EXTERN_VAR.CACHE_MAXSIZE_UNANNOTED):
    #     Get_CT_Unannotated_RAW(i)
    #     counter.increment()
    print()


def Save_CT_Candidate(index, path, data, *arg, **kwargs):
    """存储可用CT数据的缓存"""
    data = data.to(torch.device("cpu"))
    index_str = "{:06}".format(index)
    cache_file = os.path.join(path, index_str)
    joblib.dump(data, cache_file)
